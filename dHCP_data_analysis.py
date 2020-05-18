import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import glob
from os.path import expanduser
home = expanduser("~")

# read the date birth data
with open(home+'/Desktop/UPC/DD/Final_Project/Data_analysis/dhcp_t2_and_seg_data/participants.tsv', newline='') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    data = list(reader)

variables = data[0]
data.pop(0)
birth_date = []

for row in data:
    birth_date.append(row[2])

# turn the list of strings into a list of floats
birth_date_ = [float(i) for i in birth_date]

# histogram of the birth date
bins = np.linspace(min(birth_date_), max(birth_date_), 12)
histt, bin_edges = np.histogram(birth_date_, bins = bins)

fig = plt.figure(figsize=(6, 4))
plt.xlim([min(birth_date_), max(birth_date_)])

plt.hist(birth_date_, bins=bins, align='mid')
plt.title('Birth date Histogram')
plt.xlabel('Weeks')
plt.ylabel('Amount')

for i, v in enumerate(histt):
    plt.text(bins[i] + 0.5, v + 1, str(v))

# plt.show()

# cumulative histogram of the birth date
res = sc.cumfreq(birth_date_, numbins=12, defaultreallimits=(min(birth_date_), max(birth_date_)), weights=None)
bins_c = np.linspace(min(birth_date_), max(birth_date_), 12)
histt_c = np.cumsum(histt)

fig = plt.figure(figsize=(6, 4))

plt.xlim([min(birth_date_), max(birth_date_)])
plt.xticks(np.round(bins_c, 1))
plt.hist(birth_date_, bins=bins_c, align='mid', cumulative= True)
plt.title('Birth date Histogram cumulative')
plt.xlabel('Weeks')
plt.ylabel('Amount')

for i, v in enumerate(histt_c):
    plt.text(bins[i] + 0.5, v + 2, str(v))

# plt.show()

# read the mri date data
src_directory = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/dhcp_t2_and_seg_data/'

# get directories for each subject
src_subjects = glob.glob(src_directory + 'sub-*' +'/sub-*_sessions.tsv', recursive=True)

mri_date = []

for dr in src_subjects:
    with open(dr, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        data = list(reader)
        data = data[1]  # data.pop(0)
        mri_date.append(data[1])

# turn the list of strings into a list of floats
mri_date_ = [float(i) for i in mri_date]

# histogram of the mri acquisition date
bins_mri = np.linspace(min(mri_date_), max(mri_date_), 13)
histt, bin_edges = np.histogram(mri_date_, bins = bins_mri)

fig = plt.figure(figsize=(6, 4))
plt.xlim([min(mri_date_), max(mri_date_)])

plt.hist(mri_date_, bins=bins_mri, align='mid')
plt.xticks(np.round(bins_mri, 1))
plt.title('MRI acquisition Histogram')
plt.xlabel('Weeks')
plt.ylabel('Amount')

for i, v in enumerate(histt):
    plt.text(bins_mri[i] + 0.4, v + 1, str(v))

# plt.show()

# cumulative histogram of the mri acquisition date
res = sc.cumfreq(mri_date_, numbins=13, defaultreallimits=(min(mri_date_), max(mri_date_)), weights=None)
bins_mri_c = np.linspace(min(mri_date_), max(mri_date_), 13)
histt_c = np.cumsum(histt)

fig = plt.figure(figsize=(6, 4))

plt.xlim([min(mri_date_), max(mri_date_)])
plt.hist(mri_date_, bins=bins_mri_c, cumulative= True)
plt.xticks(np.round(bins_mri_c, 1))
plt.title('MRI date acquisition Histogram cumulative')
plt.xlabel('Weeks')
plt.ylabel('Amount')

for i, v in enumerate(histt_c):
    plt.text(bins_mri[i] + 0.4, v + 2.2, str(v))

plt.show()
