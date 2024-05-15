# Main file for training and testing
# author: ynie
# date: Feb, 2020

import argparse
from configs.config_utils import CONFIG
import os
import train, test, demo, demo_with_time
import numpy as np


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Total 3D Understanding.')
    parser.add_argument('config', type=str, default='configs/total3d_mgnet.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='qtrain', help='train, test, demo_with_time, demo or qtrain, qtest')
    parser.add_argument('--demo_path', type=str, default='demo/inputs/1', help='Please specify the demo path.')
    parser.add_argument('--save_results', type=str, default='demo/data_time/')
    parser.add_argument('--name', type=str, default=None, help='wandb exp name.')
    parser.add_argument('--avg_amount', type=int, default=None, help='The amount of samples to run the timing on')
    parser.add_argument('--sweep', action='store_true')
    return parser

if __name__ == '__main__':
    parser = parse_args()
    cfg = CONFIG(parser)

    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string(cfg.config)

    '''Run'''
    if cfg.config['mode'] == 'train':
        try:
            train.run(cfg)
        except KeyboardInterrupt:
            pass
        except:
            raise
        cfg.update_config(mode='test', resume=True, weight=os.path.join(cfg.save_path, 'model_best.pth'))
    if cfg.config['mode'] == 'test':
        test.run(cfg)
    if cfg.config['mode'] == 'demo':
        demo.run(cfg)
    if cfg.config['mode'] == 'demo_with_time':
        data_dict = {}
        cfg.config['mode'] = 'demo'
        for i in range(1,cfg.config["avg_amount"]+1):
            print(i)
            data = demo_with_time.run(cfg,i)   
            if data == True:
                print("No 2D detection")
            else:
                for key, value in data.items():
                    if key not in data_dict:
                        data_dict[key] = []
                    data_dict[key].append(value)
        #This makes a plot of the calculations time but the data_dict is also printed so data van be handled in other 
        #python scripts
        #print(data_dict)

        import numpy as np
        import matplotlib.pyplot as plt

        averages = {}
        for key, value in data_dict.items():
            if key != 'Start' and key != 'Total':
                averages[key] = sum(value) / len(value)

        keys = list(averages.keys())
        total = 0
        for key in keys:
            total += averages[key]

        total_average = total

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.pie(list(averages.values()), labels=keys, autopct='%1.1f%%', startangle=90)
        plt.title('Average Execution with average time '+str(total_average))

        plt.savefig("timing.png")
            

