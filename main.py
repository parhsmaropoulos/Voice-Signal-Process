import numpy as np
import os
from mfcc import MFCC_Features
from hmmlearn import hmm
import scipy.io.wavfile as wav
import pickle
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt


def predict(test_feat, trained_feat):
    predicted = []
    if isinstance(type(test_files), type([])):
        for test in test_feat:
            scores = []
            for i_number in trained_feat.keys():
                scores.append(trained_feat[i_number].decode(test))
            predicted.append(scores.index(max(scores)))
    else:
        scores = []
        for number in trained_feat.keys():
            print("score for ", number, "is", (trained_feat[number].score(test_feat)))
            scores.append(trained_feat[number].decode(test_feat))
        predicted.append(scores.index(max(scores)))
    return predicted


def train_hmm_model(features):
    print("Started Training")
    hmm_model = dict()

    for number in features.keys():
        print("Started training on number ", number)
        # Componets = 0-9 + o
        model = hmm.GMMHMM(n_components=11, n_iter=100)
        feature = np.ndarray(shape=(1, 13))
        for list_of_feats in features[number]:
            feature = np.vstack((feature, list_of_feats))
        obj = model.fit(feature)
        hmm_model[number] = obj
        print("Ended training on number ", number)
    return hmm_model


if __name__ == '__main__':

    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # folder = os.path.join(dir_path, 'recordings')
    test_files = []
    f_train = []
    l_train = []
    f_test = []
    l_test = []
    feats = dict()
    i = 0
    print("Started FE")

    # Rec lib
    # for subofold in os.listdir(folder):
    #     subfolder = os.path.join(folder, subofold)
    #     # Number
    #     # print(subofold)
    #     print("Started FE on number ", subofold)
    #     for file in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
    # #
    for file in os.listdir("./recordings"):
        # Read file and extract features
        #     filepath = os.path.join(subfolder, file)
        #     freq, audio = wav.read(filepath)
            freq, audio = wav.read("./recordings/"+file)
            mfcc_features = MFCC_Features(audio)
            # Every 5 file put 1 to the test set
            if i % 5 == 0:
                f_test.append(mfcc_features)
                l_test.append(int(file[0]))
                # l_test.append(subofold)
                test_files.append(file)
            else:
                f_train.append(mfcc_features)
                l_train.append(file[0])
                # l_train.append(subofold)
            i += 1
    print("Ended FE on number ", file[0])
    print("Ended FE")

    for i in range(0, len(f_train), len(f_train) // 10):
        feats[l_train[i]] = f_train[i: i + len(f_train) // 10]

    Train model
    hmm_model = train_hmm_model(feats)
    with open("Learned_Model.pkl", "wb") as file:
        pickle.dump(hmm_model, file)

    # Load model
    with open("Learned_Model.pkl", "rb") as file:
        hmm_model = pickle.load(file)


    # Test model

    corrects = 0
    i = 0
    for file in test_files:
        sampling_freq, signal = wav.read(file)
        #  MFCC
        mfcc_features = MFCC_Features(signal)
        number_predictions = predict(mfcc_features, hmm_model)
        number = l_test[i]
        if int(number) == number_predictions[0]:
            corrects = corrects + 1
        i += 1
        print("true: %s predicted: %i" % (number, number_predictions[0]))
    WER = ((i - corrects)/i)*100
    print("WER = ", WER, "%")
    print("corrects:", corrects)
    print("len", i)

    while True:
        # Record Voice
        fs = 44100  # Sample rate
        seconds = 16  # Duration of recording
        print("You have 16 seconds to say 4 - 10 digits")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished

        write('output1.wav', fs, myrecording)  # Save as WAV file
        # Split record

        sound_file = AudioSegment.from_wav("output1.wav")
        audio_chunks = split_on_silence(sound_file,
                                        # must be silent for at least half a second
                                        min_silence_len=500,

                                        # consider it silent if quieter than -16 dBFS
                                        silence_thresh=-30
                                        )
        if len(audio_chunks) < 4:
            print("Please say more than 4 digits!!")
            exit()
        elif len(audio_chunks) > 10:
            print("Please say less than 10 digits!!")
            exit()

        for i, chunk in enumerate(audio_chunks):
            out_file = ".\\recs\\chunk{0}.wav".format(i)
            chunk.export(out_file, format="wav")
        preds = []
        # Precitions
        for file in os.listdir('./recs'):
            print(file)
            sampling_freq, signal = wav.read("./recs/"+file)
            # MFCC
            mfcc_features = MFCC_Features(signal)
            number_predictions = predict(mfcc_features, hmm_model)
            print(number_predictions)
        #
        # for i in range(0, len(preds)):
        #     print("Estimation", preds[i])
        choise = input("Wanna try again? \n 0:No \n 1:Yes ")
        if choise == '0':
            break
