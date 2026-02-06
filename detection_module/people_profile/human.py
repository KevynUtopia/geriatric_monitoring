import os
import cv2
from PIL import Image
from scipy import stats
import torch
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torchreid
from torchreid.reid.metrics.distance import cosine_distance

import pandas as pd

def soft_action(action, confidence):
    output = {}
    for i, (act, conf) in enumerate(zip(action, confidence)):
        if act in output:
            output[act] += conf
        else:
            output[act] = conf
    return output

def threshold_action(action, confidence):
    output = []
    sit_conf, stand_conf = 0, 0
    for i, (act, conf) in enumerate(zip(action, confidence)):
        if 'drink' in act and conf > 0.1:
            output.append(('drink', 1))
        elif 'eat' in act and conf > 0.1:
            output.append(('eat', 1))
        elif 'sleep' in act and conf > 0.1:
            output.append(('sleep', 1))
        elif ('listen' in act or 'talk' in act) and conf > 0.4:
            output.append(('social', 1))
            output.append(('activity', 0))
        elif ('watch' in act) and conf > 0.4:
            output.append(('watch (TV)', 1))
            output.append(('activity', 0))
        elif 'sit' in act and conf > 0.4:
            output.append(('sit', 1))
            sit_conf = conf
        elif 'stand' in act:
            output.append(('stand', 1))
            stand_conf = conf
        elif conf > 0.4:
            output.append(('activity', 1))
        else:
            output.append(('activity', 0))

    return output, sit_conf, stand_conf





class Person:
    def __init__(self, path='', model=None, name='unknown', age=65, index=0):
        self.path = path
        self.name = name
        self.age = age
        self.index = index
        self.representation = None
        self.model = model

        self.states = {}
        self.all_states = {}
        self.skeletons = torch.empty((0, 17, 2))
        # self.template = ['drink', 'eat', 'sleep', 'social', 'sit',
        #                  'stand', 'watch (TV)'] #  'activity',
        self.template = []


    def representation_generator(self):
        path = os.listdir(self.path)
        path = [os.path.join(self.path, image) for image in path]
        if len(path) == 0:
            return False
        features = self.model.get_reid_features(path)
        features = np.mean(features, axis=0, keepdims=False)

        self.representation = features
        return True

    def say_hello(self):
        print(f'Hello, my name is {self.name} and I am {self.age} years old')

    def identify_action(self, action, confidence, soft=False):

        if not soft:
            output, sit_conf, stand_conf = threshold_action(action, confidence)
        else:
            output = soft_action(action, confidence)
            return output, 0, 0


        return output, sit_conf, stand_conf

    def add_action(self, action, timestamp):
        result_timestamp = {aspect: 0 for aspect in self.template}
        activity, confidence = [], []
        for act in action:
            act, conf = act.split('||')
            act = act.split('?')[0] if '?' in act else act
            activity.append(act)
            confidence.append(float(conf.split(')')[0]))
        result_timestamp, sit_conf, stand_conf = self.identify_action(activity, confidence, soft=True)

        self.states[timestamp] = result_timestamp

    def process_states(self):
        if len(self.states) == 0:
            return
        df = pd.DataFrame(self.states).T
        # key's col name is timestamp
        df = df.reset_index()
        df = df.rename(columns={'index': 'timestamp'})
        if len(self.all_states) == 0:
            self.all_states = df
        else:
            self.all_states = pd.concat([self.all_states, df])
        # clear self.states
        self.states = {}


    def add_skeleton(self, skeleton, pid, timestamp):
        key_frame = len(skeleton)//2
        self.skeletons = torch.cat([
                self.skeletons,
                skeleton[key_frame]['keypoints'][pid].unsqueeze(0)  # Adds dim at position 0 making it (1, 17, 2)
            ], dim=0)

    def reset(self):
        self.all_states = {}
        self.skeletons = torch.empty((0, 17, 2))



class People:
    def __init__(self, model, folder = 'people_profile/folder'):
        self.folder = folder
        self.people = {}
        self.stranger = Person()
        self.model = model

        people = os.listdir(folder)
        for person_idx in people:
            self.add_person(folder, person_idx)

        self.all_names = list(self.people.keys())
        self.all_representation = [self.people[name].representation
                                            for name in self.all_names]
        self.all_names.append('unknown')


    def get_num_people(self):
        return len(self.people)


    def add_person(self, folder, person_idx):
        person = Person(os.path.join(folder, person_idx), self.model)
        valid = person.representation_generator()
        if valid:
            print(f"adding {person_idx} ...", )
            self.people[person_idx] = person

    def find_person(self, query):
        # compare query with all features in self.people
        if isinstance(query, str):
            path = os.listdir(query)
            path = [os.path.join(query, image) for image in path]
            features = self.model(path)
        else:
            features = self.model(query)
        self.sim_dict = {}
        for idx in self.people.keys():
            sim = cosine_distance(features, self.people[idx].representation)
            self.sim_dict[idx] = torch.mean(sim).item()

        reid = min(self.sim_dict, key=self.sim_dict.get)
        values = np.array(list(self.sim_dict.values()))
        # std = np.std(values)

        min_value = np.min(values)
        other_values = [x for x in values if x != min_value]

        # Perform a one-sample t-test
        t_stat, p_value = stats.ttest_1samp(other_values, min_value)
        # print(f"found {reid} with p-value {p_value}")

        if p_value > 2e-3:
            return self.stranger, self.sim_dict[reid]
        self.people[reid].name = reid

        return self.people[reid], self.sim_dict[reid]





if __name__ == '__main__':
    # model = torchreid.models.build_model(
    #     name='osnet_x1_0',
    #     num_classes=1000,
    #     loss="softmax",
    #     pretrained=True
    # ).to("cuda")
    model = torchreid.utils.FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='/home/wzhangbu/elderlycare/weights/osnet_ain_ms_d_c.pth.tar',
            device='cuda'
        )
    people = People(model)

    query = 'people_profile/test_query'
    people.find_person(query)
