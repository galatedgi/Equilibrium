import mmap

from pydub import AudioSegment,playback
from backEnd.pyAudioAnalysis.pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as plt
from backEnd.Speaker import Speaker
from scipy.io import wavfile as wav
from playsound import playsound
import pygame
from threading import Timer
from PIL import Image

import time

from backEnd import *



def_wav_path = "../backEnd/data/wav_files/"
def_plt_path= "../backEnd/data/plot_file"
tmp_path= "../backEnd/data/tmp"

# class that stores all our code functionality
class AudioFile():

    def __init__(self,path,format,num_of_speakers):
        self.path=path
        self.format=format
        self.num_of_speakers=num_of_speakers
        self.speakers=[]
        self.audio=AudioSegment.from_wav(self.path)
        self.original_audio=AudioSegment.from_wav(self.path)


    #this tunction calls to the speaker diarization algorithm and process the audio file to determine the speakers intervals
    def process(self):
        self.cls= aS.speaker_diarization(self.path, self.num_of_speakers, lda_dim=0, plot_res=True)
        self.generate_speakers()
        self.generate_audio()
        self.create_equality_graph(0)
        self.create_equality_graph(1)

    # this function creates the speakers and assign them their own intervals in the speaker object. each speaker is a speaker object
    def generate_speakers(self):
        intervals,self.all_intervals= self.get_intervals(self.cls,self.num_of_speakers)
        for s in range(self.num_of_speakers):
            speaker=Speaker.Speaker(intervals[s])
            self.speakers.append(speaker)

    # this function distributes the speakers and assign them their own intervals according to the classifier given by the speaker diarizatioin algotirhm.
    def get_intervals(self,cls, num_of_speakers):
        intervals = {}
        all_intervals = []
        for i in range(num_of_speakers):
            intervals[i] = []
        t = 0
        while t < len(cls):
            speaker = cls[t]
            strat = t
            while t < len(cls) and cls[t] == speaker:
                t += 1
            intervals[speaker].append((strat * 0.2, t * 0.2))
            all_intervals.append((strat * 0.2, (t) * 0.2))
        return intervals, all_intervals

    #saving an audio file as wav
    def save_file(self, path):
        self.audio.export(path, "wav")

    #saving graph as image
    def save_fig(self,path,new_path):
        im1 = Image.open(path)
        im1.save(new_path)

    #this is an equalize function. equalize all speakers sound to the highest speaker sound
    def multi_fix_sound(self):
        sound = self.original_audio
        speakersLoudness = {}
        for speaker in range(self.num_of_speakers):
            speakersLoudness[speaker] = []
        for i in range(len(self.speakers)):
            intervals=self.speakers[i].get_intervals()
            for interval in intervals:
                part = sound[interval[0] * 1000:interval[1] * 1000]
                speakersLoudness[i].append(part.dBFS)
        averages = []
        for i in range(self.num_of_speakers):
            tempAvg = sum(speakersLoudness[i]) / len(speakersLoudness[i])
            averages.append(tempAvg)
        additions = []
        maxSound = max(averages)
        for i in range(self.num_of_speakers):
            additions.append(maxSound - averages[i])


        for i in range(self.num_of_speakers):
            self.speakers[i].set_vol(additions[i])
        self.generate_audio()

    #this function take the current volume parameters and creates ans save a temporary file with the changes.
    def generate_audio(self):
        additions=[speaker.get_vol() for speaker in self.speakers]
        self.all_seg_vol_additions(additions)
        self.audio.export(tmp_path+"/tmp_audio.wav","wav")

    # given an array of values this function adds those values to the different speakers in order to amplify them
    def all_seg_vol_additions(self,additions):
        def is_part_of_speaker(segments, tu):
            for seg in segments:
                if seg[0] > tu[0]:
                    return False
                if seg[0] == tu[0] and seg[1] == tu[1]:
                    return True
        newsong = AudioSegment.empty()
        for interval in self.all_intervals:
            for i in range(self.num_of_speakers):
                if is_part_of_speaker(self.speakers[i].get_intervals(), interval):
                    newsong = newsong + (self.original_audio[interval[0] * 1000:interval[1] * 1000] + additions[i])
        self.audio=newsong

    #this function returns the first sound segment of a given speaker
    def get_speaker_example(self,i):
        speaker=self.speakers[i]
        interval=speaker.get_intervals()[0]
        speaker_example=self.original_audio[interval[0] * 1000:interval[1] * 1000]+speaker.get_vol()
        return speaker_example

    #this function create the graph of the speakers voice and the time.
    def create_equality_graph(self,type):
        plt.clf()
        plt.xlabel("time")
        plt.ylabel("frequency")
        if type==0:#original audio
            rate, data1 = wav.read(self.path)
            time1 = [x / 46000 for x in range(len(data1))]
            plt.plot(time1,data1)
            plt.savefig(tmp_path+"/original_plot.png")
        elif type==1:
            rate, data1 = wav.read(tmp_path+"/tmp_audio.wav")
            time1 = [x / 46000 for x in range(len(data1))]
            plt.plot(time1,data1)
            plt.savefig(tmp_path+"/equality_plot.png")


    #getter to a specific speaker current volume addition
    def get_speaker_vol(self,i):
        return self.speakers[i].get_vol()

    #setter to a specific speaker current volume addition
    def set_speaker_vol(self,i,vol):
        return self.speakers[i].set_vol(vol)

    #play the last audio file that was generated
    def play(self):
        pygame.init()
        song = pygame.mixer.Sound(tmp_path+"/tmp_audio.wav")
        clock = pygame.time.Clock()
        song.play()
        t = Timer(song.get_length(), pygame.quit)
        t.start()

    #function to stop the current audio play
    def stop(self):
        pygame.quit()

    #
    def get_all_intervals(self):
        return self.all_intervals

    #function to play a segment from a specific speaker
    def play_speaker_segment(self,index):
        audio=self.get_speaker_example(index)
        audio.export(tmp_path+"/speaker.wav","wav")
        pygame.init()
        song = pygame.mixer.Sound(tmp_path+"/speaker.wav")
        clock = pygame.time.Clock()
        song.play()
        t = Timer(song.get_length(), pygame.quit)
        t.start()

    def get_speakers(self):
        return self.speakers


