from people_profile import Person
from tqdm import tqdm
from utils import update_time
import pandas as pd
import os
import torch

class Analyzor:
    def __init__(self):

        self.humans = {}


    def update_states(self, reid, action, skeleton, keys, start_time=''):
        """
            action, reid, skeleton are all dict of timestamps
            At each timestamp, i.e., action[timestamp], reid[timestamp], skeleton[timestamp]:
                action[timestamp] (List):       [ num_pid * [action||likelihood] ]

                reid[timestamp] (List):         [ num_pid * {'name', 'box_int'} ]

                skeleton[timestamp] (List):     [ num_frame * { 'keypoints', 'keypoint_scores} ]
                        'keypoints' in shape of [num_pid, 17, 2],
                        'keypoint_scores' in shape of [num_pid, 17]

            keys (List):    [timestamp1, timestamp2, ...]

            start_time (str):  the start time of the video, in format of hhmmss
        """

        for frame in (keys):
            identities = reid[frame]
            actions = action[frame]
            skeletons = skeleton[frame]
            if identities is None or actions is None or skeletons is None:
                continue


            for i, (idx, act) in enumerate(zip(identities, actions)):
                name = idx['name'].split('PID:')[-1]
                if 'unknown' in name:
                    continue
                if name not in self.humans:
                    person = Person(name=name)
                    self.humans[name] = person
                else:
                    person = self.humans[name]

                # start_time is in format of hhmmss, second is second
                # get the updated time
                updated_time = update_time(start_time, frame//15)
                # if name=='p_2':
                #     print(frame)
                    # print(f"Processing {name} at {updated_time} of frame {frame}...")
                person.add_action(act, updated_time)
                # for skeleton in skeletons:
                person.add_skeleton(skeletons, pid=i, timestamp=updated_time)
        for p in self.humans.keys():
            self.humans[p].process_states()




    def analyze(self):
        people = list(self.humans.keys())
        print(people)
        print(self.humans[people[0]].states)

    def save_results(self, out_dir):

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print(f"Saving results to {out_dir} ...")
        for p in self.humans.keys():
            person_state = self.humans[p].all_states
            person_state.to_csv(os.path.join(out_dir, f"{p}.csv"), index=False)
            skeletons = self.humans[p].skeletons
            torch.save(skeletons, os.path.join(out_dir, f"{p}_skeleton.pt"))
        self.humans = {}