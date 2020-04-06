"""
audio
(frame,26) => (block of frames,26,64)
motion
(frame,192) => (block of frames,192,64)

motion's example: (100, 192) =>padding=> (128,192) =>reshape=> (2,192,64)

※motionは標準化
"""

import sys
import os
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import pyquaternion as pyq
import copy
from sklearn import preprocessing

from python_speech_features import mfcc

from tools import average, extract_prosodic_features, shorten, calculate_spectrogram

N_OUTPUT = 192 * 2 # Number of gesture features (position) and label
FEATURES = "MFCC"
DATA_DIR = sys.argv[1]

if FEATURES == "MFCC":
    N_INPUT = 26 # Number of MFCC features

MFCC_INPUTS=26 # How many features we will store for each MFCC vector

def create_hierarchy_nodes(hierarchy):
    """
    Create hierarchy nodes: an array of markers used in the motion capture
    Args:
        hierarchy: bvh file read in a structure

    Returns:
        nodes: array of markers to be used in motion processing

    """
    joint_offsets = []
    joint_names = []

    for idx, line in enumerate(hierarchy):
        hierarchy[idx] = hierarchy[idx].split()
        if not len(hierarchy[idx]) == 0:
            line_type = hierarchy[idx][0]
            if line_type == 'OFFSET':
                offset = np.array([float(hierarchy[idx][1]), float(hierarchy[idx][2]), float(hierarchy[idx][3])])
                joint_offsets.append(offset)
            elif line_type == 'ROOT' or line_type == 'JOINT':
                joint_names.append(hierarchy[idx][1])
            elif line_type == 'End':
                joint_names.append('End Site')

    nodes = []
    for idx, name in enumerate(joint_names):
        if idx == 0:
            parent = None
        elif idx in [6, 30]: #spine1->shoulders
            parent = 2
        elif idx in [14, 18, 22, 26]: #lefthand->leftfingers
            parent = 9
        elif idx in [38, 42, 46, 50]: #righthand->rightfingers
            parent = 33
        elif idx in [54, 59]: #hip->legs
            parent = 0
        else:
            parent = idx - 1

        if name == 'End Site':
            children = None
        elif idx == 0: #hips
            children = [1, 54, 59]
        elif idx == 2: #spine1
            children = [3, 6, 30]
        elif idx == 9: #lefthand
            children = [10, 14, 18, 22, 26]
        elif idx == 33: #righthand
            children = [34, 38, 42, 46, 50]
        else:
            children = [idx + 1]

        node = dict([('name', name), ('parent', parent), ('children', children), ('offset', joint_offsets[idx]), ('rel_degs', None), ('abs_qt', None), ('rel_pos', None), ('abs_pos', None)])
        if idx == 0:
            node['rel_pos'] = node['abs_pos'] = [float(0), float(60), float(0)]
            node['abs_qt'] = pyq.Quaternion()
        nodes.append(node)

    return nodes

def rot_vec_to_abs_pos_vec(frames, nodes):
    """
    Transform vectors of the human motion from the joint angles to the absolute positions
    Args:
        frames: human motion in the join angles space
        nodes:  set of markers used in motion caption

    Returns:
        output_vectors : 3d coordinates of this human motion
    """
    output_lines = []

    for frame in frames:
        node_idx = 0
        for i in range(51): #changed from 51
            stepi = i*3
            z_deg = float(frame[stepi])
            x_deg = float(frame[stepi+1])
            y_deg = float(frame[stepi+2])

            if nodes[node_idx]['name'] == 'End Site':
                 node_idx = node_idx + 1
            nodes[node_idx]['rel_degs'] = [z_deg, x_deg, y_deg]
            current_node = nodes[node_idx]

            node_idx = node_idx + 1

        for start_node in nodes:
            abs_pos = np.array([0, 60, 0])
            current_node = start_node
            if start_node['children'] is not None: #= if not start_node['name'] = 'end site'
                for child_idx in start_node['children']:
                    child_node = nodes[child_idx]

                    child_offset = np.array(child_node['offset'])
                    qz = pyq.Quaternion(axis=[0, 0, 1], degrees=start_node['rel_degs'][0])
                    qx = pyq.Quaternion(axis=[1, 0, 0], degrees=start_node['rel_degs'][1])
                    qy = pyq.Quaternion(axis=[0, 1, 0], degrees=start_node['rel_degs'][2])
                    qrot = qz * qx * qy
                    offset_rotated = qrot.rotate(child_offset)
                    child_node['rel_pos']= start_node['abs_qt'].rotate(offset_rotated)

                    child_node['abs_qt'] = start_node['abs_qt'] * qrot

            while current_node['parent'] is not None:

                abs_pos = abs_pos + current_node['rel_pos']
                current_node = nodes[current_node['parent']]
            start_node['abs_pos'] = abs_pos

        line = []
        for node in nodes:
            line.append(node['abs_pos'])
        output_lines.append(line)

    output_vels = []
    for idx, line in enumerate(output_lines):
        vel_line = []
        for jn, joint_pos in enumerate(line):
           if idx == 0:
               vels = np.array([0.0, 0.0, 0.0])
           else:
               vels = np.array([joint_pos[0] - output_lines[idx-1][jn][0], joint_pos[1] - output_lines[idx-1][jn][1], joint_pos[2] - output_lines[idx-1][jn][2]])
           vel_line.append(vels)
        output_vels.append(vel_line)

    out = []
    for idx, line in enumerate(output_vels):
        ln = []
        for jn, joint_vel in enumerate(line):
            ln.append(output_lines[idx][jn])
            ln.append(joint_vel)
        out.append(ln)

    output_array = np.asarray(out)
    output_vectors = np.empty([len(output_array), N_OUTPUT])
    for idx, line in enumerate(output_array):
        output_vectors[idx] = line.flatten()
    return output_vectors

def calculate_mfcc(audio_filename):
    """
    Calculate MFCC features for the audio in a given file
    Args:
        audio_filename: file name of the audio

    Returns:
        feature_vectors: MFCC feature vector for the given audio file
    """
    fs, audio = wav.read(audio_filename)

    # Make stereo audio being mono
    if len(audio.shape) == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    # Calculate MFCC feature with the window frame it was designed for
    input_vectors = mfcc(audio, winlen=0.02, winstep=0.01, samplerate=fs, numcep=MFCC_INPUTS)

    input_vectors = [average(input_vectors[:, i], 5) for i in range(MFCC_INPUTS)]

    feature_vectors = np.transpose(input_vectors)

    return feature_vectors

def create_vectors(audio_filename, output_file, nodes):
    """
    Extract features from a given pair of audio and motion files
    Args:
        audio_filename:    file name for an audio file (.wav)
        gesture_filename:  file name for a motion file (.bvh)
        nodes:             an array of markers for the motion

    Returns:
        input_vectors   : speech features
        output_vectors  : motion features
    """
    input_vectors = calculate_mfcc(audio_filename)

    f = open(output_file, 'r')
    org = f.readlines()
    frametime = org[310].split()

    del org[0:311]

    bvh_len = len(org)

    for idx, line in enumerate(org):
        org[idx] = [float(x) for x in line.split()]

    for i in range(0, bvh_len):
        for j in range(0, int(306 / 3)):
            st = j * 3
            del org[i][st:st + 3]

    if float(frametime[2]) == 0.0416667:
        del org[::6]
    elif float(frametime[2]) == 0.010000:
        org = org[::5]
    else:
        print("smth wrong with fps of " + output_file)

    output_vectors = rot_vec_to_abs_pos_vec(org, nodes)

    f.close()

    input_vectors, output_vectors = shorten(input_vectors, output_vectors)

    # 速度データを削除
    for i in range(64):
        output_vectors = np.delete(output_vectors,[3*i+3,3*i+4,3*i+5],axis=1)

    # 後でセンテンスを分ける時に使うパラメータ
    length = len(output_vectors)

    return input_vectors,output_vectors, length

def reshapedata(X,Y):
    remainder=64-len(X)%64
    input_vectors,output_vectors = pad_sequence(X,Y,remainder)
    input_divided,output_divided = divided(input_vectors,output_vectors)

    return input_divided,output_divided

def pad_sequence(input_vectors,output_vectors,remainder):
    """
    Pad array of features in order to be able to take context at each time-frame
    We pad N_CONTEXT / 2 frames before and after the signal by the features of the silence
    Args:
        input_vectors:      feature vectors for an audio

    Returns:
        new_input_vectors:  padded feature vectors
    """

    # Pad sequence not with zeros but with MFCC of the silence
    silence_vectors = calculate_mfcc("./silence.wav")
    mfcc_empty_vector = silence_vectors[0]
    empty_vectors = np.array([mfcc_empty_vector] * int(remainder))

    # output_vectors
    motion_empty_vector = np.zeros([1,192])
    empty_output_vectors = np.array([motion_empty_vector[0]] * int(remainder))
    new_output_vectors = np.append(output_vectors, empty_output_vectors, axis=0)

    new_input_vectors = np.append(input_vectors, empty_vectors, axis=0)

    return new_input_vectors, new_output_vectors

def divided(inputs,outputs):
    reminder = int(len(inputs)/64)
    input_divided = np.reshape(inputs,(reminder,64,26))
    output_divided = np.reshape(outputs,(reminder,64,192))

    input_divided_tr = np.array([])
    output_divided_tr = np.array([])
    for num,(input_s,output_m) in enumerate(zip(input_divided,output_divided)):
        input_s_tr = np.transpose(input_s)
        output_m_tr = np.transpose(output_m)
        if num == 0:
            input_divided_tr = np.reshape(input_s_tr,[1,26,64])
            output_divided_tr = np.reshape(output_m_tr,[1,192,64])
        else:
            input_divided_tr = np.append(input_divided_tr,input_s_tr.reshape(1,26,64),axis=0)
            output_divided_tr = np.append(output_divided_tr,output_m_tr.reshape(1,192,64),axis=0)
    return input_divided_tr,output_divided_tr

def create_test_sequences(name, nodes):
    print(name)
    DATA_FILE = pd.read_csv(DATA_DIR + '/gg-' + str(name) + '.csv')
    X = np.array([])
    Y = np.array([])

    for i in range(len(DATA_FILE)):
        input_vectors, output_vectors,length = create_vectors(DATA_FILE['wav_filename'][i], DATA_FILE['bvh_filename'][i], nodes)
        input_s, output_m = reshapedata(input_vectors,output_vectors)
        filenumber1 = DATA_FILE['bvh_filename'][i].split("/")[-1]
        filenumber = filenumber1.split(".")[0]
        x_file_name = './test_inputs/X_'+str(name)+'_'+str(filenumber)+'_posegan.npy'
        print(DATA_FILE['bvh_filename'][i])
        print("X datasize",input_s.shape)
        np.save(x_file_name, input_s)
    return

def main(name, nodes):
    print(name)
    DATA_FILE = pd.read_csv(DATA_DIR + '/gg-' + str(name) + '.csv')
    X = np.array([])
    Y = np.array([])

    # 全センテンスの音声データはX、モーションデータはYに保存
    for i in range(len(DATA_FILE)):
        input_vectors, output_vectors,length = create_vectors(DATA_FILE['wav_filename'][i], DATA_FILE['bvh_filename'][i], nodes)
        if len(X) == 0:
            X = input_vectors
            Y = output_vectors
            Leng = np.array([length])
        else:
            X = np.concatenate((X, input_vectors), axis=0)
            Y = np.concatenate((Y, output_vectors), axis=0)
            Leng = np.append(Leng, length+Leng[-1])
        if i%3==0:
            print("^^^^^^^^^^^^^^^^^^")
            print('{:.2f}% of processing for {:.8} dataset is done'.format(100.0 * (i+1) / len(DATA_FILE), str(name)))
            print("Current dataset sizes are:")
            print(X.shape)
            print(Y.shape)

    # Yは標準化
    Y_z = preprocessing.scale(Y)
    ave = np.mean(Y,axis=0)
    std = np.std(Y,axis=0)

    # 再度センテンスごとに分ける
    X_s = np.split(X,Leng) 
    Y_s = np.split(Y_z,Leng) 
    del X_s[-1]
    del Y_s[-1]
    X_out = np.array([])
    Y_out = np.array([])
    for X_s1,Y_s1 in zip(X_s,Y_s):
        # センテンスごとに分け、padding&divide
        # motion: (frames, 192) => (block of frames, 192, 64)
        # speech: (frames, 26) => (block of frames, 26, 64)
        input_s, output_m = reshapedata(X_s1,Y_s1)
        if len(Y_out) == 0:
            X_out = np.array(input_s,dtype=float)
            Y_out = output_m
        else:
            X_out = np.concatenate((X_out, input_s), axis=0)
            Y_out = np.concatenate((Y_out, output_m), axis=0)

    normalization_path = "./norm/" 
    os.makedirs(normalization_path, exist_ok=True)

    ave_name = normalization_path+'ave_'+str(name)+'_posegan.npy'
    std_name = normalization_path+'std_'+str(name)+'_posegan.npy'
    np.save(ave_name, ave)
    np.save(std_name, std)
    x_file_name = './X_'+str(name)+'_posegan.npy'
    y_file_name = './Y_'+str(name)+'_posegan_norm.npy'
    print("X datasize",X_out.shape,"Y datasize",Y_out.shape)
    np.save(x_file_name, X_out)
    np.save(y_file_name, Y_out)


if __name__ == "__main__":
    # Check if script get enough parameters
    if len(sys.argv) < 2:
        raise ValueError('Not enough paramters! \nUsage : python ' + sys.argv[0].split("/")[-1] + ' DATA_DIR')

    # Check if the dataset exists
    if not os.path.exists(sys.argv[1]):
        raise ValueError(
            'Path to the dataset ({}) does not exist!\nPlease, provide correct DATA_DIR as a script parameter'
            ''.format(sys.argv[1]))

    f = open('hierarchy.txt', 'r')
    hierarchy = f.readlines()
    f.close()
    nodes = create_hierarchy_nodes(hierarchy)

    namelist = ["dev","train"]
    for name in namelist:
        main(name,nodes)

    # test_inputs
    create_test_sequences("test",nodes)
    print("--complete--")