import os
import sys
home_path = os.path.expanduser('~')
sys.path.append(os.path.join(home_path, 'elderlycare'))
import pickle
import glob
from analyze_module import Analyzor
from get_args import alignment_parse_args
from tqdm import tqdm
import pandas as pd



class Alignment_Worker:
    def __init__(self, all_cam, all_candidates):
        self.all_candidates = all_candidates
        self.all_views = {}
        self.all_cam = all_cam
        self.all_time_consistency = {}
        self.all_person_align = {} # {persone_name: {cam_name: [time1:{}, time2:{}, ...]}}
        self.states = {}

    def initialize(self, input_path):

        for person in self.all_candidates:

            print(f"Processing {person} ...")
            self.output_path = os.path.join(input_path, 'alignment', person + '.csv')

            all_view = {}
            time_consistency = []

            ####
            for cam in self.all_cam:
                view = os.path.join(input_path, cam, person + '.csv')
                if os.path.exists(view):
                    # print(f"Processing {self.all_cam[0]} ...")
                    view = pd.read_csv(view, header=0, index_col=0)
                    all_view[cam] = view
                    time_consistency.extend(view.index.tolist())
            ####
            # view0 = os.path.join(input_path, 'cam_10', person + '.csv')
            # if os.path.exists(view0):
            #     # print(f"Processing {self.all_cam[0]} ...")
            #     view0 = pd.read_csv(view0, header=0, index_col=0)
            #     all_view[self.all_cam[0]] = view0
            #     time_consistency.extend(view0.index.tolist())
            #
            # view1 = os.path.join(input_path, 'cam_11', person + '.csv')
            # if os.path.exists(view1):
            #     # print(f"Processing {self.all_cam[1]} ...")
            #     view1 = pd.read_csv(view1, header=0, index_col=0)
            #     all_view[self.all_cam[1]] = view1
            #     time_consistency.extend(view1.index.tolist())
            #
            # view2 = os.path.join(input_path, 'cam_12', person + '.csv')
            #
            # if os.path.exists(view2):
            #     # print(f"Processing {self.all_cam[2]} ...")
            #     view2 = pd.read_csv(view2, header=0, index_col=0)
            #
            #     all_view[self.all_cam[2]] = view2
            #     time_consistency.extend(view2.index.tolist())
            #
            # view3 = os.path.join(input_path, 'cam_13', person + '.csv')
            # if os.path.exists(view3):
            #     # print(f"Processing {self.all_cam[3]} ...")
            #     view3 = pd.read_csv(view3, header=0, index_col=0)
            #
            #     all_view[self.all_cam[3]] = view3
            #     time_consistency.extend(view3.index.tolist())

            self.all_views[person] = all_view

            time_consistency.sort()
            result = [time_consistency[0]]
            for num in time_consistency[1:]:
                if abs(num - result[-1]) <= 3:
                    # Keep the first occurrence (or use result[-1] = num to keep last)
                    continue
                else:
                    result.append(num)
            self.all_time_consistency[person] = set(result)

    def align(self):
        for person in self.all_candidates:

            print(f"Aligning {person} ...")
            # self.all_views contains all persons' all_view, all_view contains all cameras' view
            all_view = self.all_views[person]
            all_view_align = {}

            for v in all_view.keys():
                view = all_view[v]
                view_time = list(set(view.index) & self.all_time_consistency[person])


                for time in view_time:
                    # save all results to the same all_view_align to aggregare all views
                    df = pd.DataFrame(view.loc[[time]])
                    if time in all_view_align:
                        all_view_align[time] = pd.concat([all_view_align[time], df], verify_integrity=False)
                    else:
                        all_view_align[time] = df



            self.all_person_align[person] = all_view_align

    def post_process(self, output_path, task='alignment', soft=False):
        output_path = os.path.join(output_path, task)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Create folder {output_path}")

        for person in self.all_candidates:

            # self.all_viewse contains all persons' all_view, all_view contains all cameras' view
            all_view_align = self.all_person_align[person]
            for time in all_view_align.keys():
                try:
                    df = pd.DataFrame(all_view_align[time])
                except ValueError:
                    df = pd.concat(all_view_align[time])
                except:
                    print(f"Error: {person} {time}")
                    continue

                if soft:
                    # status = df.groupby(level=0).sum()
                    status = df.groupby(level=0).mean()
                else:
                    status = df.mode().apply(lambda x: x.max())
                    status.name = df.index[0]
                    status = status.to_frame().T
                if person in self.states:
                    self.states[person] = pd.concat([self.states[person], status])
                else:
                    self.states[person] = status


            self.states[person] = self.states[person].sort_index()
            # remove noises
            self.states[person] = self.states[person][self.states[person].index > 10000]
            print(f"Saving {person} ...")
            self.states[person].to_csv(os.path.join(output_path, f"{person}.csv"), index=True)





if __name__ == '__main__':
    args = alignment_parse_args()

    # recording_2019_06_24_8_05_am
    all_cam = os.listdir(args.input_path)

    all_cam = [cam for cam in all_cam if cam.startswith('cam_')]
    all_cam.sort()

    all_cam_candidates = {}

    for cam in all_cam:
        all_files = os.listdir(os.path.join(args.input_path, cam))
        # find all files end with .csv as postfix, and remove that postfix
        candidates = [os.path.splitext(file)[0] for file in all_files if file.endswith('.csv')]
        candidates.sort()
        all_cam_candidates[cam] = candidates


    # get the union of all candidates
    all_candidates = set(all_cam_candidates[all_cam[0]])
    for cam in all_cam[1:]:
        all_candidates = all_candidates.union(set(all_cam_candidates[cam]))
    all_candidates = list(all_candidates)
    all_candidates.sort()

    align = Alignment_Worker(all_cam=all_cam, all_candidates=all_candidates)
    align.initialize(input_path=args.input_path)
    align.align()
    align.post_process(output_path=args.output_path, task=args.task, soft=args.soft)

    '''
    recording_2019_06_25_8_00_am  recording_2019_06_29_8_00_am  recording_2019_07_13_6_50_am  
    recording_2019_07_18_6_50_am  recording_2019_07_6_7_50_am   recording_2019_06_26_8_15_am   
    recording_2019_07_10_6_50_am  recording_2019_07_15_6_50_am  recording_2019_07_3_6_50_am   
    recording_2019_07_8_6_50_am   recording_2019_06_22_9_20_am  recording_2019_06_27_10_30_am  
    recording_2019_07_11_6_50_am  recording_2019_07_16_7_20_am  recording_2019_07_4_6_50_am   
    recording_2019_07_9_6_50_am   recording_2019_06_24_8_05_am  recording_2019_06_28_7_55_am   
    recording_2019_07_12_6_50_am  recording_2019_07_17_7_20_am  recording_2019_07_5_7_50_am
    '''




