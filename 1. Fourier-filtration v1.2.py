#!/usr/bin/env python
# coding: utf-8

# In[2]:


#1. Фурье-фильтрация сигнала.
#• Смоделировать сигнал из трёх некратных частот, добавить случайный шум.
#• Вычислить спектр сигнала, нарисовать график, оси в реальных единицах (Гц).
#• Избавиться от шума в исходном сигнале путём вырезания “лишних” частот в фурьеспектре.
#• Построить график исходного, зашумлённого и фильтрованного сигнала с легендой.


from numpy.random import sample
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq, irfft
from numpy import cos, pi, arange, abs as nabs, where, mean, array, zeros


# In[11]:


#• Смоделировать сигнал из трёх некратных частот, добавить случайный шум.

FD = 500
N = 50000
filter_line = 3

A = 3; m1 = 1.4; w1 = 5; phi1 = 5; w2 = 40; phi2 = 1;
m2 = 0.2;

t = arange(N)/FD
signal1 = abs( A*(1 + m1*cos(2*pi*w1*t + phi1))*cos(2*pi*w2*t + phi2) )
signal2 = abs( A*(1 + m2*cos(2*pi*w1*t + phi1))*cos(2*pi*w2*t + phi2) )

#noise = (-6)*sample(signal.shape[0]) + 3 #слишком большая амплитуда шума: (-np.pi, np.pi), хороший шум: (-0.1, 0.1)

#noised_signal = signal + noise


# In[19]:


#• Вычислить спектр сигнала, нарисовать график, оси в реальных единицах (Гц).

freq = rfftfreq(N, 1./FD)
clear_ampl1 = 2*nabs(rfft(signal1))/N
clear_ampl2 = 2*nabs(rfft(signal2))/N
#ampl = 2*nabs(rfft(noised_signal))/N
#spectrum = rfft(noised_signal)

#print(len())

fig = plt.figure(figsize = (10, 10))
sub = fig.add_subplot(111) #цифры внутри - соотножения сторон от фигсайз

#plt.plot(freq, ampl, label = 'Frequences', c = 'green')
#plt.plot(freq, [mean(ampl) for i in range(freq.shape[0])], c = 'orange')
#plt.plot(freq, [filter_line*mean(ampl) for i in range(freq.shape[0])], c = 'red')
plt.plot(freq, clear_ampl1, c = 'blue')
plt.plot(freq, clear_ampl2, c = 'red')

sub.set_xlabel('Частоты (Гц)', fontsize = 12)
sub.set_ylabel('Напряжение (B)', fontsize = 12)

plt.legend(fontsize = 12)
plt.grid(True)
plt.show()

# In[47]:


#• Избавиться от шума в исходном сигнале путём вырезания “лишних” частот в фурьеспектре.
    #пройдём по массиву, возьмем те частоты, при которых рост сменяется спадом
    #возьмем их амплитуды
    #по ним будем делать обратное преобразование Фурье
    #лучше просто убрать маленькие частоты по условию

#первый проход, убираем шум по линии от средней амплитуды
#filtrated_frequances = [freq[i] if ampl[i] > filter_line*mean(ampl) else 0 for i in range(freq.shape[0])]
#filtrated_signal = [spectrum[i] if ampl[i] > filter_line*mean(ampl) else 0 for i in range(freq.shape[0])]

#Остатки можно почистить ~так: из последовательных данных взять только те частоты которые соответсвуют смене роста на убыль амплитуды

#filtrated_frequances
#можно попробовать всё, что ниже среднего + 18% убирать; Больше 18% получилось - ~30%


# In[48]:


#fig = plt.figure(figsize = (10, 10))
#sub = fig.add_subplot(111) #цифры внутри - соотножения сторон от фигсайз

#plt.plot(t, noised_signal, label = 'Зашумленный сигнал', c = 'orange')
#plt.plot(t, irfft(filtrated_signal, N), label = 'Фильтрованный сигнал', c = 'green')
#plt.plot(t, signal, label = 'Чистый сигнал', c = 'blue')

#sub.set_xlabel('Время (с)', fontsize = 12)
#sub.set_ylabel('Напряжение (В)', fontsize = 12)
#sub.set_title('График исходного, зашумлённого и фильтрованного сигнала')

#plt.legend(loc = 'upper center', fontsize = 12)
#plt.grid(True)


# In[15]:


#signal.shape[0]


# In[249]:


#import numpy

#help(np.shape)
#help(plt.figure().add_subplot())
#help(plt.plot)
#help(plt.legend)
#help(numpy.random.uniform)


# In[564]:


#что-то бесполезное но в один момент работаютщее

    #беда с моим исполнением избирания "хороших" частот в том, что я знаю сколько хороших частот
    #беда в том, что я передавал хороший сигнал и думал, что это так и должно работать


#for i in range (0, 10): #тут можно оценивать количество главных частот, но это костыль
        #where(ampl == max(ampl))[0][0] #в каждой ячейке лежит число и с ним тип
#wm = where(ampl == max(ampl))[0][0]
#freq[wm]
    #filtrated_frequances[wm] = freq[wm]
    #filtrated_signal[wm] = spectrum[wm]
        #ampl - max(ampl) это опускает весь сигнал на одну высоту, максимум просто меняет модуль на столько
        #нужно по этому индексу поставить ноль:
    #filtrated_amplitude[wm] = max(ampl)
    #ampl[where(ampl == max(ampl))] = 0
    #занулять амплитуды - хуйня, по амплитудам нужно восстанавливать амлитуды функций
        #ampl[where(ampl == max(ampl))[0][0]] просто пишет max(ampl) :)
#filtrated_amplitude
