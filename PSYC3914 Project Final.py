#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats.stats import pearsonr   
import seaborn as sns
import copy
import platform
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Get the operating system name
os_name = platform.system()
# Check if it's macOS or Windows
if os_name == "Darwin":
    os = "mac"
elif os_name == "Windows":
    os = 'Windows'
else:
    os = os_name
    print(f"You are using {os_name}.")
print(os_name)


# In[3]:


if os == "mac":
    filepath = r'/Users/macbookpro15touchbar/Library/CloudStorage/OneDrive-Personal/Docs Sync/Jobs And Money/Careers/Study/Uni/USyd/2024S2/PSYC3914/PSYC3914 Assessments/Research Assignment/PSYC3914 Project Data Class.csv'
if os == "Windows":
    filepath = r'E:\oDrive\OneDrive\Docs Sync\Jobs And Money\Careers\Study\Uni\USyd\2024S2\PSYC3914\PSYC3914 Assessments\Research Assignment\\PSYC3914 Project Data Class.csv'
df = pd.read_csv(filepath)

# Some small changes made to the data
# Inconsistencies in gender harmonised (M/F/Male/Female/male/female) 
# 3+ languages spoken recoded to 3
# "Which language/s are you fluent in?" broken up in to separate data columns


# In[4]:


#Tiktok
df = df.rename(columns={'How many hours per day do you spend on short video content (e.g. TikTok) on average?': 'tiktok'})
df = df.rename(columns={'Consider your use of short form video content (e.g. TikTok, Instagram Reels, Youtube Shorts). Approximately how long have you been using these platforms in MONTHS?': 'tiktok_time'})
df = df.rename(columns={'Do you consume short form content more often than you wish to?': 'tiktok_addict'})
df = df.rename(columns={'How many hours per day do you spend on your mobile phone? ': 'phone_time'})


# In[5]:


#Speed
df = df.rename(columns={'Consider your engagement in watching recorded lectures. Do you increase the speed of these videos (e.g. to 1.25x, 1.5x, 2x speed)?': 'speed'})


# In[6]:


# Games
df = df.rename(columns={'Do you play chess?': 'chess'})
df = df.rename(columns={'How many hours do you spend playing video games per day?': 'vgames'})


# In[7]:


# Stress, calming
df = df.rename(columns={'On a scale of 0-10, how stressed did you feel on the day/s you completed these tests?': 'stress'})
df = df.rename(columns={'How often do you use specific techniques (such as journaling, deep breathing, or talking to someone) to regulate your emotions?': 'emotion_reg'})
df = df.rename(columns={'Before beginning each TMB test, did you take at least 10 seconds to steady your thoughts or breathing?': 'preparation'})


# In[8]:


#Music
df = df.rename(columns={'On average, how many hours of music do you listen to per day?': 'music'})
df = df.rename(columns={'Would you say you prefer the aesthetic qualities (e.g. pleasing to listen to, energising/relaxing) or thematic qualities (e.g. lyrical meaning, emotional experience) of music?': 'music_type'})
df = df.rename(columns={'How many genres of music do you listen to?': 'music_num_genres'})
df = df.rename(columns={'Do you currently play any musical instruments?': 'music_play_instr'})
#Reading
df = df.rename(columns={'How many hours do you spend reading non-university related texts/novels per week?': 'reading'})


# In[9]:


#Enjoyment
df = df.rename(columns={'How much did you enjoy doing the Matrix Reasoning task?    Recall that the task required you to select the shape that best completes the pattern.': 'enjoy_matrix'})


# In[10]:


#Openness, creative
df = df.rename(columns={'Do you like trying new experiences, even if you do not know how they will turn out?': 'new_experiences'})
df = df.rename(columns={'Do you see yourself as a creative person?': 'creative'})


# In[11]:


#Exercise
df = df.rename(columns={'How many hours do you engage in physical activity per week?': 'exercise_hours'})
df = df.rename(columns={'How would you rate your lifestyle in terms of physical activity?': 'exercise_rating'})
#Sleep
df = df.rename(columns={'On average, how many hours of sleep do you get per night?': 'sleep_hours'})
df = df.rename(columns={'How would you rate your sleep quality overall?': 'sleep_quality'})

#Mood
df = df.rename(columns={'Rate your mood at the time of completing these tests  ': 'mood'})


# In[12]:


#Languages
df = df.rename(columns={'Is English your first language?': 'languages_eng_1st'})
df = df.rename(columns={'How many languages do you speak?': 'languages_num'})
df = df.rename(columns={'What is the language you are second-most confident with?': 'languages_second'})
#df = df.rename(columns={'Which language/s are you fluent in?': 'languages_fluent'})


# In[13]:


#Demography
df = df.rename(columns={'How many siblings do you have?': 'siblings'})


# In[14]:


df_recoded = copy.deepcopy(df)


# In[15]:


df.columns.values


# In[16]:


#Recodings
df_recoded = copy.deepcopy(df)
df_recoded['languages_num'] = df_recoded['languages_num'].replace("1 (English only)", '1') #
df_recoded['languages_num'] = df_recoded['languages_num'].replace("3+", '3')

df_recoded['speed'][(df['speed'] == "No")] = 0
df_recoded['speed'][(df['speed'] == "Yes")] = 1

df_recoded['Gender '][(df['Gender '] == "Female")] = 0
df_recoded['Gender '][(df['Gender '] == "Genderqueer")] = 0.5 #####
df_recoded['Gender '][(df['Gender '] == "Male")] = 1
df_recoded['Handedness '][(df['Handedness '] == "Left")] = 0
df_recoded['Handedness '][(df['Handedness '] == "Ambidextrous")] = 0.5 ######
df_recoded['Handedness '][(df['Handedness '] == "Right")] = 1

df_recoded['preparation'][(df_recoded['preparation'] == "No")] = 0
df_recoded['preparation'][(df_recoded['preparation'] == "Yes")] = 1

## Chess
# Binary measure (play chess: yes/no)
df_recoded['chess_any'] = df['chess']
df_recoded['chess_any'][(df_recoded['chess'] == "Yes, occasionally") | (df_recoded['chess'] == "Yes, often")] = 1
df_recoded['chess_any'][(df_recoded['chess'] == "No")] = 0
# Continuous measure
df_recoded['chess_graded'] = df['chess']
df_recoded['chess_graded'][(df_recoded['chess'] == "No")] = 0
df_recoded['chess_graded'][(df_recoded['chess'] == "Yes, occasionally")] = 1
df_recoded['chess_graded'][(df_recoded['chess'] == "Yes, often")] = 2


df_recoded['emotion_reg'][(df_recoded['emotion_reg'] == "Never")] = 0
df_recoded['emotion_reg'][(df_recoded['emotion_reg'] == "Sometimes")] = 1
df_recoded['emotion_reg'][(df_recoded['emotion_reg'] == "Always")] = 2

df_recoded['languages_eng_1st'][(df['languages_eng_1st'] == "No")] = 0
df_recoded['languages_eng_1st'][(df['languages_eng_1st'] == "Yes")] = 1
df_recoded['music_play_instr'][(df['music_play_instr'] == "No, never")] = 0
df_recoded['music_play_instr'][(df['music_play_instr'] == "No, but did in past")] = 1
df_recoded['music_play_instr'][(df['music_play_instr'] == "Yes, leisurely")] = 2
df_recoded['music_play_instr'][(df['music_play_instr'] == "Yes, professionally")] = 3
df_recoded['music_num_genres'][(df['music_num_genres'] == "4+")] = 4
df_recoded['music_type'][(df['music_type'] == "Aesthetic")] = 0
df_recoded['music_type'][(df['music_type'] == "Thematic")] = 1
df_recoded['tiktok_addict'][(df['tiktok_addict'] == "No")] = 0
df_recoded['tiktok_addict'][(df['tiktok_addict'] == "Yes")] = 1

df_recoded['new_experiences'][(df['new_experiences'] == "No")] = 0
df_recoded['new_experiences'][(df['new_experiences'] == "Yes")] = 1
df_recoded['creative'][(df['creative'] == "No")] = 0
df_recoded['creative'][(df['creative'] == "Yes")] = 1


# In[17]:


## Combination variables # composite # combined
df_recoded['tiktok_cumulative'] = df_recoded["tiktok_time"]*df_recoded['tiktok']
df_recoded['sleep_overall'] = df_recoded['sleep_hours']*df_recoded['sleep_quality']
df_recoded['digit_span'] = df_recoded['TMB Forward Digit Span']+df_recoded['TMB Backward Digit Span']
df_recoded['multiracial'] = df_recoded['TMB Multiracial Emotion Identification'] + df_recoded['TMB Multiracial Reading the Mind in the Eyes']
df_recoded['trail_making_avg'] = (df_recoded['TMB Trail-Making (A)'] + df_recoded['TMB Trail-Making (B)'])*0.5

#WAIS-IV uses these
iq_variables = ['digit_span', # 'TMB Forward Digit Span', 'TMB Backward Digit Span'
                'TMB Matrix Reasoning',
                #'TMB Vocabulary',
                'TMB Verbal Paired Associates Memory - Test','TMB Visual Paired Associates Memory - Test',
                'trail_making_avg', #'TMB Trail-Making (A)','TMB Trail-Making (B)',
                'TMB Paced Serial Addition'
               ]


## Weighting
# Make all IQ variables count the same
sum_ = 0
for var in iq_variables:
    sum_+= df_recoded[var].sum()
avg_ = sum_ / len(iq_variables)

#Make all TMB variables count the same
sum_all = 0
p = 0
for var in df_recoded.columns.values:
    if "TMB" in var:
        sum_all+= df_recoded[var].sum()
        p+=1
avg_all = sum_all / p

df_recoded['iq'] = 0.0
df_recoded['iq_equalweight'] = 0.0
df_recoded['iq_normalised'] = 0.0
for var in iq_variables:
    ## Method to use normalised scores
    array = df_recoded[var]
    df_recoded[var+"_normalised"] = array

    mean = np.nanmean(df_recoded[var])
    std = np.nanstd(df_recoded[var])
    # Normalize the array (convert to z-scores), while ignoring NaNs
    normalized_array = (array - mean) / std
    df_recoded[var+"_normalised"] = normalized_array
    # Calculagte stdev diff
    df_recoded['iq_normalised'] = df_recoded['iq_normalised'] + df_recoded[var+"_normalised"]    
    
    ## Method to use average
    df_recoded['iq'] = df_recoded['iq'] + df_recoded[var]
    df_recoded[var+"_equal"] = df_recoded[var]*(avg_/df_recoded[var].sum())
    df_recoded['iq_equalweight'] = df_recoded['iq_equalweight'] + df_recoded[var+"_equal"]

for var in iq_variables:    
    del_cols = ["iq",'iq_equalweight',var+"_equal",var+"_normalised"]
    df_recoded = df_recoded.drop(del_cols, axis=1, errors='ignore')

# Take average normalised score
df_recoded['iq_normalised'] = df_recoded['iq_normalised'] / len(iq_variables) 


# Use ALL TMB variables to contribute to IQ calculation
# df_recoded['iq_tmb_all'] = 0.0
# df_recoded['iq_tmb_equalweight'] = 0.0
# for var in df_recoded.columns.values:
#     if "TMB" in var:
#         df_recoded[var+"_tmb_equal"] = df_recoded[var]*(avg_all/df_recoded[var].sum())
#         df_recoded['iq_tmb_all'] = df_recoded['iq_tmb_all'] + df_recoded[var]        
#         df_recoded['iq_tmb_equalweight'] = df_recoded['iq_tmb_all'] + df_recoded[var+'_tmb_equal']


# In[18]:


#df_recoded[['iq','iq_equalweight',"iq_normalised"]].head(5)
df_recoded[["iq_normalised"]].head(5)


# In[19]:


#Cols to delete
del_cols = ["languages_second",
            #"How would you rate your overall proficiency in this second language, including speaking, reading, and writing? ",
            "Which language/s are you fluent in?",
            "Unnamed: 25",
            "Unnamed: 27",
            "Unnamed: 28",
           ]

del_cols =  del_cols + ["Participant ID",
            "Age (years)",
             "Gender ",
             "Handedness ",
             "siblings"
            ]
df_recoded = df_recoded.drop(del_cols, axis=1, errors='ignore')


# In[20]:


# Show frequency of responses for each question with a few categories
for col in df_recoded.columns.values:
    # Count the frequency of each value
    counted_data = Counter(df_recoded[col].values)
    num_results = len(counted_data)
    if num_results < 5:
        sorted_counted_data = counted_data.most_common()
        # Display the results
        print(col)
        for item, count in sorted_counted_data:
            print(f"{item}: {count}")
        print("\n")


# In[21]:


### # Function to compute Pearson correlation and test for significance
def correlation_significance_test(x, y, alpha=0.05):
    # Calculate the Pearson correlation coefficient and the p-value
    mask = ~np.isnan(x) & ~np.isnan(y) 
    x_filtered = x[mask]
    y_filtered = y[mask]

    r, p_value = stats.pearsonr(x_filtered, y_filtered)
    
    # Determine if the correlation is significant based on the p-value
    if p_value < alpha:
        print(f"The correlation is statistically significant (r = {r:.2f}, p = {p_value:.3f}).")
    else:
        print(f"The correlation is not statistically significant (r = {r:.2f}, p = {p_value:.3f}).")
    
    return r, p_value


def plotter(xlabel,ylabel):
    plt.figure()
    x = df_recoded[xlabel]
    y = df_recoded[ylabel]

    mask = ~np.isnan(x) & ~np.isnan(y) 
    x_filtered = x[mask]
    y_filtered = y[mask]

    plt.scatter(x_filtered, y_filtered, color='blue', marker='o')
    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Calculate the line of best fit
    coefficients = np.polyfit(x_filtered, y_filtered, 1)  # 1 indicates a linear fit
    poly = np.poly1d(coefficients)
    y_fit = poly(x_filtered)
    # Plot the line of best fit
    plt.plot(x_filtered, y_fit, color='red', linestyle='--')
    plt.title(ylabel+" vs "+str(xlabel))
    # Show the plot
    plt.show()




# In[22]:


plotting = 0
# Plot the relationship between all the significant pairs, where one is the "find_var" argument
def plot_all(find_var):
    for elem in significant_pairs:
        try:
            if elem[0] == find_var or elem[1] == find_var:
                pass
            if elem[0] == find_var:
                print(elem[1], significant_pairs[elem])
                if plotting == 1:
                    plotter(elem[0],elem[1])
            elif elem[1] == find_var:
                print(elem[0], significant_pairs[elem])
                if plotting == 1:
                    plotter(elem[0],elem[1])
        except:
            print("Problem with :", elem[0], "and ", elem[1])
            
def is_array_numeric(arr):
    return np.issubdtype(arr.dtype, np.number)


# In[23]:


for col in df_recoded.columns.values:
    try:
        df_recoded[col] = pd.to_numeric(df_recoded[col], errors='coerce')
    except:
        print("Error", col)
    #print(col,is_array_numeric(df_recoded[col].values),df_recoded[col].head(5))


# In[24]:


# #Plot the distribution for each variable
# for col in df_recoded.columns.values:
#     try:
#         print(col)

#         # Sample data
#         data = df_recoded[col].values
    
#         plt.figure()
#         # Create a figure with subplots
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
#         # Scatter plot
#         ax1.scatter(np.arange(len(data)), data, color='blue')
#         ax1.set_title('Scatter Plot of Data Points')
#         ax1.set_xlabel('Index')
#         ax1.set_ylabel('Value')
        
#         # Box plot
#         sns.boxplot(data=data, ax=ax2, color='lightgreen')
#         ax2.set_title('Box Plot of Data')
        
#         # Adjust layout for better visualization
#         plt.tight_layout()
        
#         # Show the plots
#         plt.show()
#     except:
#         print("Error with col", col)


# In[25]:


# Replace erroneous user input data 

# Error values
df_recoded.loc[39, 'tiktok'] = np.nan # # someone said they spend 30 hours per day on tiktok! Maybe they meant per week but I'm just going to remove them
df_recoded.loc[39, 'tiktok_cumulative'] = np.nan # is altered by the above error since it's a multiple of it
#Inspect the data

# # People who don't use at all
# df_recoded.loc[2, 'tiktok'] = np.nan
# df_recoded.loc[19, 'tiktok'] = np.nan
# df_recoded.loc[25, 'tiktok'] = np.nan
# # High users
# df_recoded.loc[21, 'tiktok'] = np.nan
# df_recoded.loc[28, 'tiktok'] = np.nan
# df_recoded.loc[29, 'tiktok'] = np.nan
# df_recoded.loc[36, 'tiktok'] = np.nan


#Then remove the biggest outlier
# df_recoded.loc[26, 'tiktok'] = np.nan # 4 # 4 hours/day??
#df1.loc[26, "TMB Forward Digit Span"] = np.nan # 7


df_recoded.loc[30, 'TMB Simple Reaction Time'] = np.nan # Implausibly low reaction time
# Scores above the maximum possible, no obvious data entry error that I can correct
df_recoded.loc[18, 'TMB Verbal Paired Associates Memory - Test'] = np.nan
df_recoded.loc[30, 'TMB Verbal Paired Associates Memory - Test'] = np.nan
df_recoded.loc[6, 'TMB Visual Paired Associates Memory - Test'] = np.nan
df_recoded.loc[25, 'TMB Visual Paired Associates Memory - Test'] = np.nan
df_recoded.loc[22, 'music'] = np.nan # Does not listen to 20hrs of  music per day


# In[26]:


# Hypothesis 1
# We expect a negative correlation between time watching videos and each of the following:
# Digit Span (both Forward Digit Span, and Backward Digit Span)
### Tiktok variable to use for analyses
tiktok_var_to_use_dct = {1: 'tiktok', 2: 'tiktok_time', 3: 'tiktok_cumulative', 4: "tiktok_addict"}
tiktok_var_to_use = tiktok_var_to_use_dct[1]
if tiktok_var_to_use == 'tiktok':  
    des = 'tiktok usage (hrs/day)' 
    ttl = 'tiktok usage'
elif tiktok_var_to_use == 'tiktok_time':
    des = 'tiktok usage duration (months)'
    ttl = 'tiktok usage'
elif tiktok_var_to_use == 'tiktok_cumulative':
    des = 'cumultative tiktok usage (~month-hours)'
    ttl = 'tiktok usage'
elif tiktok_var_to_use == 'tiktok_addict':
    des = 'tiktok addict status'
    ttl = 'tiktok addiction'    
print(des)


# In[27]:


#Preview 
df1 = df_recoded[["tiktok","tiktok_time","tiktok_cumulative","tiktok_addict","TMB Forward Digit Span","TMB Backward Digit Span","digit_span","stress"]]
df1_corr = df1.corr()
df1_corr.round(2)


# In[28]:


# Do those who report consuming more short video content than they wish to consume more?
df_addict_no = df1[df1['tiktok_addict'] == 0]
addict_no = df_addict_no[tiktok_var_to_use].values
df_addict_yes = df1[df1['tiktok_addict'] == 1]
addict_yes = df_addict_yes[tiktok_var_to_use].values

mask = ~np.isnan(addict_no)
addict_no = addict_no[mask]
mask = ~np.isnan(addict_yes)
addict_yes = addict_yes[mask]


# Perform independent two-sample t-test
t_statistic, p_value = stats.ttest_ind(addict_no, addict_yes, equal_var=False)  # Set equal_var=False for Welch's t-test


# Calculate variances and sample sizes
s1_squared = np.var(addict_no, ddof=1)
s2_squared = np.var(addict_yes, ddof=1)
n1 = len(addict_no)
n2 = len(addict_yes)

# Calculate Welch-Satterthwaite degrees of freedom
df = ((s1_squared/n1 + s2_squared/n2)**2) / ((s1_squared**2 / ((n1**2) * (n1-1))) + (s2_squared**2 / ((n2**2) * (n2-1))))

# Output the results
print(f"T-statistic: {t_statistic}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value}")


    
# Compute the standard deviation (sample)
std_dev = np.std(addict_no, ddof=1)  # ddof=1 for sample standard deviation
# Compute the sample size
n2 = len(addict_no)
# Compute the standard error
standard_error_nonaddict = std_dev / np.sqrt(n2)

# Compute the standard deviation (sample)
std_dev = np.std(addict_yes, ddof=1)  # ddof=1 for sample standard deviation
# Compute the sample size
n2 = len(addict_yes)
# Compute the standard error
standard_error_addict = std_dev / np.sqrt(n2)

print("mean, std_error for non-addicts: ", np.mean(addict_no),standard_error_nonaddict)
print("mean, std_error for addicts: ", np.mean(addict_yes),standard_error_addict)


# In[29]:


# Sample data for x and y
x = df1["TMB Forward Digit Span"]
y = df1["TMB Backward Digit Span"]
z = df1[tiktok_var_to_use]


# In[30]:


# Create a plot
# Filter out NaN values from both x and y
mask = ~np.isnan(x) & ~np.isnan(y) 
x_filtered = x[mask]
y_filtered = y[mask]

plt.figure()
plt.scatter(x_filtered, y_filtered, color='blue', marker='o')
# Add labels and title
plt.xlabel('Forward digit span', fontweight='bold')
plt.ylabel('Backward digit span', fontweight='bold')
# Calculate the line of best fit
coefficients = np.polyfit(x_filtered, y_filtered, 1)  # 1 indicates a linear fit
print(coefficients)
poly = np.poly1d(coefficients)#(x)
y_fit = poly(x_filtered)
# Plot the line of best fit
plt.plot(x_filtered, y_fit, color='red', linestyle='--')
plt.title('Forward vs backward digit span')
# Show the plot
plt.show()


# Example: enter your two arrays
x_ = np.array(x_filtered)
y_ = np.array(y_filtered)
# Perform the significance test
correlation_significance_test(x_, y_)


# In[31]:


# Is digit span different in low vs high stress individuals?
# Calculate the xth percentile of the specified column
low = 0.25
high = 1 - low
percentile_low = df1['stress'].quantile(low)
percentile_high = df1['stress'].quantile(high)
# Filter the DataFrame for rows where the column's value is in the top 25th percentile
filtered_lowstress = df1[df1['stress'] <= percentile_low]
filtered_lowstress_digitspans = filtered_lowstress["TMB Forward Digit Span"].values
filtered_highstress = df1[df1['stress'] >= percentile_high]
filtered_highstress_digitspans = filtered_highstress["TMB Forward Digit Span"].values
# Perform independent two-sample t-test
t_statistic, p_value = stats.ttest_ind(filtered_lowstress_digitspans, filtered_highstress_digitspans, equal_var=False)  # Set equal_var=False for Welch's t-test

# Output the results
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")
# Calculate sample variances
s1_squared = np.var(filtered_lowstress_digitspans, ddof=1)
s2_squared = np.var(filtered_highstress_digitspans, ddof=1)
n1 = len(filtered_lowstress_digitspans)
n2 = len(filtered_highstress_digitspans)
# Calculate degrees of freedom for Welch's t-test
dof = ((s1_squared / n1) + (s2_squared / n2))**2 / \
     (((s1_squared / n1)**2 / (n1 - 1)) + ((s2_squared / n2)**2 / (n2 - 1)))
print(f"Degrees of Freedom: {dof}")

# Compute the standard deviation (sample)
std_dev = np.std(filtered_lowstress_digitspans, ddof=1)  # ddof=1 for sample standard deviation
# Compute the sample size
n2 = len(filtered_lowstress_digitspans)
# Compute the standard error
standard_error_lowstress = std_dev / np.sqrt(n2)

# Compute the standard deviation (sample)
std_dev = np.std(filtered_highstress_digitspans, ddof=1)  # ddof=1 for sample standard deviation
# Compute the sample size
n2 = len(filtered_highstress_digitspans)
# Compute the standard error
standard_error_highstress = std_dev / np.sqrt(n2)

print("mean, std_error for low stress forward digit span: ", np.mean(filtered_lowstress_digitspans),standard_error_lowstress)
print("mean, std_error for high stress forward digit span: ", np.mean(filtered_highstress_digitspans),standard_error_highstress)


# In[32]:


#Low stress individuals - correlation between digit span and tiktok usage
x_lowstress = filtered_lowstress["TMB Forward Digit Span"]
y_lowstress = filtered_lowstress["TMB Backward Digit Span"]
z_lowstress = filtered_lowstress[tiktok_var_to_use]

# Create a plot
# Filter out NaN values from both x and y
mask = ~np.isnan(x_lowstress) & ~np.isnan(z_lowstress) 
x_filtered_lowstress = x_lowstress[mask]
z_filtered_lowstress = z_lowstress[mask]

plt.figure()
plt.scatter(x_filtered_lowstress, z_filtered_lowstress, color='blue', marker='o')
# Add labels and title
plt.xlabel('Forward digit span')
plt.ylabel(des)
# Calculate the line of best fit
coefficients = np.polyfit(x_filtered_lowstress, z_filtered_lowstress, 1)  # 1 indicates a linear fit
print(coefficients)
poly = np.poly1d(coefficients)#(x)
z_fit = poly(x_filtered_lowstress)
# Plot the line of best fit
plt.plot(x_filtered_lowstress, z_fit, color='red', linestyle='--')
plt.title('Forward digit span vs '+ttl+" for low stress people")
# Show the plot
plt.show()

# Example: enter your two arrays
x_ = np.array(x_filtered_lowstress)
z_ = np.array(z_filtered_lowstress)
# Perform the significance test
correlation_significance_test(x_, z_)


# In[33]:


# High stress individuals - correlation between digit span and tiktok usage
x_highstress = filtered_highstress["TMB Forward Digit Span"]
y_highstress = filtered_highstress["TMB Backward Digit Span"]
z_highstress = filtered_highstress[tiktok_var_to_use]

# Create a plot
# Filter out NaN values from both x and y
mask = ~np.isnan(x_highstress) & ~np.isnan(z_highstress) 
x_filtered_highstress = x_highstress[mask]
z_filtered_highstress = z_highstress[mask]

plt.figure()
plt.scatter(x_filtered_highstress, z_filtered_highstress, color='blue', marker='o')
# Add labels and title
plt.xlabel('Forward digit span')
plt.ylabel(des)
# Calculate the line of best fit
coefficients = np.polyfit(x_filtered_highstress, z_filtered_highstress, 1)  # 1 indicates a linear fit
print(coefficients)
poly = np.poly1d(coefficients)#(x)
z_fit = poly(x_filtered_highstress)
# Plot the line of best fit
plt.plot(x_filtered_highstress, z_fit, color='red', linestyle='--')
plt.title('Forward digit span vs '+ttl+" for high stress people")
# Show the plot
plt.show()

# Example: enter your two arrays
x_ = np.array(x_filtered_highstress)
z_ = np.array(z_filtered_highstress)
# Perform the significance test
correlation_significance_test(x_, z_)


# In[34]:


t = df1[tiktok_var_to_use]


# In[35]:


for var in ["TMB Forward Digit Span","TMB Backward Digit Span"]:
    print("Variable to compare to tiktok use is: ", var)
    y = df1[var]
    # Create a plot
    # Filter out NaN values from both x and y
    mask = ~np.isnan(t) & ~np.isnan(y) 
    t_filtered = t[mask]
    y_filtered = y[mask]
    
    plt.figure()
    plt.scatter(t_filtered, y_filtered, color='blue', marker='o')
    mpl.rcParams['font.family'] = 'Arial'
    # Add labels and title
    #plt.xlabel(des)
    #plt.ylabel(var)
    plt.xlabel("Short-video consumption (months)", fontsize=14, color='black', fontweight='bold')  # X-axis label custom size and color
    plt.ylabel(var, fontsize=12, color='black', fontweight='bold', rotation=90)  # Y-axis label custom size and color and rotate vertically
    plt.ylabel("Backward Digit Span", fontsize=14, color='black', fontweight='bold', rotation=90)  # Y-axis label custom size and color and rotate vertically
    
    
    # Calculate the line of best fit
    coefficients = np.polyfit(t_filtered, y_filtered, 1)  # 1 indicates a linear fit
    poly = np.poly1d(coefficients)#(x)
    y_fit = poly(t_filtered)
    # Plot the line of best fit
    plt.plot(t_filtered, y_fit, color='red', linestyle='--')
    plt.title(var+" vs "+ttl)
    # Show the plot
    plt.show()
    
    # Example: enter your two arrays
    t_ = np.array(t_filtered)
    y_ = np.array(y_filtered)
    # Perform the significance test
    correlation_significance_test(t_, y_)


# In[36]:


for var in ["TMB Forward Digit Span","TMB Backward Digit Span"]:
    print("Variable to compare to tiktok use is: ", var)
    y = df1[var]
    # Create a plot
    # Filter out NaN values from both x and y
    mask = ~np.isnan(t) & ~np.isnan(y) 
    t_filtered = t[mask]
    y_filtered = y[mask]
    
    plt.figure()
    plt.scatter(t_filtered, y_filtered, color='blue', marker='o')
    # # Add labels and title
    # plt.xlabel(des)
    # plt.ylabel(var)
    # Calculate the line of best fit
    coefficients = np.polyfit(t_filtered, y_filtered, 1)  # 1 indicates a linear fit
    poly = np.poly1d(coefficients)#(x)
    y_fit = poly(t_filtered)
    # Plot the line of best fit
    plt.plot(t_filtered, y_fit, color='red', linestyle='--')
    # Show the plot
    mpl.rcParams['font.family'] = 'Arial'
    # Customizing the title, x-axis, and y-axis
    plt.title(var+" vs "+des, fontsize=14, color='black')  # Custom font size and jet black title
    #plt.title("Forward Digit Span"+" vs "+"short-video consumption", fontsize=14, color='black')  # Custom font size and jet black title
    plt.xlabel(des, fontsize=12, color='black')  # X-axis label custom size and color
    plt.xlabel("Short-video consumption (months)", fontsize=14, color='black', fontweight='bold')  # X-axis label custom size and color
    plt.ylabel(var, fontsize=12, color='black', rotation=90)  # Y-axis label custom size and color and rotate vertically
    plt.ylabel("Forward Digit Span", fontsize=14, color='black', fontweight='bold', rotation=90)  # Y-axis label custom size and color and rotate vertically
    
    # Customize the tick labels font size and color
    plt.tick_params(axis='x', labelsize=12, labelcolor='black')  # X-axis tick labels size and color
    plt.tick_params(axis='y', labelsize=12, labelcolor='black')  # Y-axis tick labels size and color
    plt.show()
    
    # Example: enter your two arrays
    t_ = np.array(t_filtered)
    y_ = np.array(y_filtered)
    # Perform the significance test
    correlation_significance_test(t_, y_)


# In[37]:


for var in ["TMB Forward Digit Span","TMB Backward Digit Span"]:
    print("Variable to compare to tiktok use is: ", var)
    y = df1[var]
    # Create a plot
    # Filter out NaN values from both x and y
    mask = ~np.isnan(t) & ~np.isnan(y) 
    t_filtered = t[mask]*365.25/12
    y_filtered = y[mask]
    
    plt.figure()
    plt.scatter(t_filtered, y_filtered, color='blue', marker='o')
    # Calculate the line of best fit
    coefficients = np.polyfit(t_filtered, y_filtered, 1)  # 1 indicates a linear fit
    poly = np.poly1d(coefficients)#(x)
    y_fit = poly(t_filtered)
    # Plot the line of best fit
    plt.plot(t_filtered, y_fit, color='red', linestyle='--')
    

    mpl.rcParams['font.family'] = 'Arial'
    # Customizing the title, x-axis, and y-axis
    plt.title(var+" vs "+ttl)
    plt.title("Forward Digit Span"+" vs "+"short-video consumption", fontsize=14, color='black')  # Custom font size and jet black title
    #plt.xlabel(des, fontsize=12, color='black')  # X-axis label custom size and color
    plt.xlabel("Cumulative short-video consumption (hours)", fontsize=12, fontweight = 'bold', color='black')  # X-axis label custom size and color
    #plt.ylabel(var, fontsize=12, color='black', rotation=90)  # Y-axis label custom size and color and rotate vertically
    plt.ylabel("Forward Digit Span", fontsize=12, fontweight = 'bold', color='black', rotation=90)  # Y-axis label custom size and color and rotate vertically

    # Show the plot
    plt.show()
    
    # Example: enter your two arrays
    t_ = np.array(t_filtered)
    y_ = np.array(y_filtered)
    # Perform the significance test
    correlation_significance_test(t_, y_)


# In[38]:


# Hypotheses 2
# We expect a negative correlation between time watching videos and each of the following:
# TMB Paced Serial Addition","TMB Gradual Onset Continuous Performance Test"
df2 = df_recoded[["TMB Paced Serial Addition","TMB Gradual Onset Continuous Performance Test","tiktok","tiktok_cumulative","tiktok_time"]]
df2_corr = df2.corr()
df2_corr.round(2)


# In[39]:


###### Remove outliers
use_outliers = 0
if use_outliers == 0:
    df2.loc[25, "TMB Paced Serial Addition"] = np.nan # biggest outlier
    # df2.loc[26, "TMB Paced Serial Addition"] = np.nan 
    # df2.loc[13, "TMB Paced Serial Addition"] = np.nan 
    # df2.loc[36, "TMB Paced Serial Addition"] = np.nan
else:
    df2.loc[25, "TMB Paced Serial Addition"] = 38 # iggest outlier
    df2.loc[26, "TMB Paced Serial Addition"] = 37
    df2.loc[13, "TMB Paced Serial Addition"] = 41
    df2.loc[36, "TMB Paced Serial Addition"] = 46


# In[40]:


#Inspect the data


# Sample data for x and y
x = df2["TMB Paced Serial Addition"]
y = df2["TMB Gradual Onset Continuous Performance Test"]
z = df2[tiktok_var_to_use] 

# Create a plot
# Filter out NaN values from both x and y
mask = ~np.isnan(x) & ~np.isnan(z) 
x_filtered = x[mask]
z_filtered = z[mask]

plt.figure()
#Flip axes
x_ = x_filtered[0:]
x_filtered = z_filtered[0:]
z_filtered = x_[0:]


plt.scatter(x_filtered, z_filtered, color='blue', marker='o')

# Add labels and title
plt.xlabel('Paced Serial Addition')
plt.ylabel(des)

# # Calculate the line of best fit
coefficients = np.polyfit(x_filtered, z_filtered, 1)
# Generate the line of best fit for the entire x-range
slope, intercept = coefficients
x_line = np.linspace(np.nanmin(x_filtered), np.nanmax(x_filtered), 100)  # Use the full x-range, ignoring NaNs
y_line = slope * x_line + intercept
# Plot the original data (filtered)
plt.scatter(x_filtered, z_filtered, color='blue', marker='o', label='Data points')
# Plot the line of best fit over the full x-range
plt.plot(x_line, y_line, color='red', linestyle='--', label='Line of best fit')

# Show the plot
mpl.rcParams['font.family'] = 'Arial'
# Customizing the title, x-axis, and y-axis
plt.title('Paced Serial Addition vs '+ttl)
plt.title('Paced Serial Addition vs '+"short-video consumption", fontsize=14, color='black')  # Custom font size and jet black title
plt.xlabel(des)
plt.xlabel("Short-video consumption (hours/day)", fontsize=12, fontweight='bold',color='black')  # X-axis label custom size and color
plt.ylabel('Paced Serial Addition')
plt.ylabel('Paced Serial Addition', fontsize=12, fontweight='bold', color='black', rotation=90)  # Y-axis label custom size and color and rotate vertically

# Customize the tick labels font size and color
plt.tick_params(axis='x', labelsize=12, labelcolor='black')  # X-axis tick labels size and color
plt.tick_params(axis='y', labelsize=12, labelcolor='black')  # Y-axis tick labels size and color


# Show the plot
plt.show()


# Example: enter your two arrays
x_ = np.array(x_filtered)
z_ = np.array(z_filtered)
# Perform the significance test
correlation_significance_test(x_, z_)


# In[41]:


# Create a plot
# Filter out NaN values from both x and y
mask = ~np.isnan(y) & ~np.isnan(z) 
y_filtered = y[mask]
z_filtered = z[mask]

plt.figure()
plt.scatter(y_filtered, z_filtered, color='blue', marker='o')
# Add labels and title
plt.xlabel('Gradual Onset Continuous Performance')
plt.ylabel(des)


# # Calculate the line of best fit
coefficients = np.polyfit(y_filtered, z_filtered, 1)
slope, intercept = coefficients
# Generate the line of best fit for the entire x-range
x_line = np.linspace(np.nanmin(y_filtered), np.nanmax(y_filtered), 100)  # Use the full x-range, ignoring NaNs
y_line = slope * x_line + intercept
# Plot the original data (filtered)
plt.scatter(y_filtered, z_filtered, color='blue', marker='o', label='Data points')
# Plot the line of best fit over the full x-range
plt.plot(x_line, y_line, color='red', linestyle='--', label='Line of best fit')

plt.title('Gradual Onset Continuous Performance vs '+ttl)
plt.xlabel('Paced Serial Addition')
plt.ylabel(des)
# Show the plot
plt.show()


# Example: enter your two arrays
y_ = np.array(y_filtered)
z_ = np.array(z_filtered)
# Perform the significance test
correlation_significance_test(y_, z_)


# In[42]:


### Hypothesis 3A
# More chess playing should have lead to better Paced Serial Attention 

df3 = df_recoded.copy()

# Binary chess vs non-chess vs chess_graded
for var in ["TMB Forward Digit Span","TMB Backward Digit Span", "TMB Matrix Reasoning", 
            "TMB Paced Serial Addition", "TMB Digit Symbol Matching",
            "TMB Verbal Paired Associates Memory - Test", "TMB Visual Paired Associates Memory - Test"]:
    
    print("Variable to compare to chess use is: ", var)
    
    df3[var+"_nochess"] = (df3[var])*(df3["chess_any"]==0)
    df3[var+"_chess"] = (df3[var])*(df3["chess_any"]==1)
    df3[var+"_chess"] = (df3[var])*(df3["chess_graded"]==2)
    data_nochess = df3[var+"_nochess"].values
    data_chess = df3[var+"_chess"].values
    filtered_data_nochess = data_nochess[(data_nochess != 0) & ~np.isnan(data_nochess)]
    filtered_data_chess = data_chess[(data_chess != 0) & ~np.isnan(data_chess)]


    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_nochess, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n1 = len(filtered_data_nochess)
    # Compute the standard error
    standard_error_nochess = std_dev / np.sqrt(n1)
    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_chess, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n2 = len(filtered_data_chess)
    # Compute the standard error
    standard_error_chess = std_dev / np.sqrt(n2)

    mean_a = np.mean(filtered_data_nochess)
    se_a = standard_error_nochess
    mean_b = np.mean(filtered_data_chess)
    se_b = standard_error_chess    
    print("Without chess: ", mean_a , se_a )
    print("With chess: ", mean_b , se_b )

    # Calculate sample variances
    s1_squared = np.var(filtered_data_nochess, ddof=1)
    s2_squared = np.var(filtered_data_chess, ddof=1)
    
    # Calculate degrees of freedom for Welch's t-test
    dof = ((s1_squared / n1) + (s2_squared / n2))**2 / \
         (((s1_squared / n1)**2 / (n1 - 1)) + ((s2_squared / n2)**2 / (n2 - 1)))
    
    print(f"Degrees of Freedom: {dof}")

    # Perform independent two-sample t-test
    t_statistic, p_value = stats.ttest_ind(filtered_data_nochess, filtered_data_chess, equal_var=False)  # Set equal_var=False for Welch's t-test
    
    # Output the results
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")


    
    # Data: the two numbers and their respective standard errors
    numbers = [mean_a, mean_b]  # The two numbers to compare
    std_errors = [se_a, se_b]  # Standard errors for each number

    # X-axis labels for the bars
    labels = ['No chess', 'Frequent chess']
    
    # Define the positions for the bars on the x-axis
    x_pos = np.arange(len(numbers))
    
    # Create the bar chart
    fig, ax = plt.subplots()
    mpl.rcParams['font.family'] = 'Arial'
    bars = ax.bar(x_pos, numbers, yerr=std_errors, align='center', alpha=0.7, capsize=10, color=['blue', 'orange'])
    
    # Add labels, title, and customizations
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    #ax.set_ylabel('Value')
    ax.set_title('Comparison of Two Numbers with Standard Error Bars')
    
    plt.xlabel("Chess activity", fontsize=14, color='black', fontweight='bold')  # X-axis label custom size and color
    plt.ylabel(var, fontsize=14, color='black', fontweight='bold', rotation=90)  # Y-axis label custom size and color and rotate vertically
    
    
    # Display the plot
    plt.show()

    print("\n")



# In[43]:


# Hypothesis 3
# Students who play chess may also do better than non-chess-players (as per (Aciego et al., 2012)
#...because chess also leads to implicit use of chunking (Thalmann et al., 2019) 
df3_b_corr = df3[["TMB Paced Serial Addition","chess_any","chess_graded"]]


# In[44]:


# Hypotheses 4
# We expect positive correlations between time watching videos and the following:
# -	Multiracial Emotion Identification
# -	Multiracial Reading the Mind in the Eye
# -	The Cambridge Face Memory Test
df4 = df_recoded[["TMB Multiracial Emotion Identification","TMB Multiracial Reading the Mind in the Eyes","multiracial","TMB Cambridge Face Memory Test","tiktok","tiktok_time","tiktok_cumulative"]]
df4_corr = df4.corr()
df4_corr.round(2)


# In[45]:


t = df4[tiktok_var_to_use]

for var in ["TMB Multiracial Emotion Identification","TMB Multiracial Reading the Mind in the Eyes","multiracial","TMB Cambridge Face Memory Test"]:
    print("Variable to compare to tiktok use is: ", var)
    
    y = df4[var]
    # Create a plot
    # Filter out NaN values from both x and y
    mask = ~np.isnan(t) & ~np.isnan(y) 
    t_filtered = t[mask]
    y_filtered = y[mask]
    
    plt.figure()
    plt.scatter(t_filtered, y_filtered, color='blue', marker='o')
    # Add labels and title
    plt.xlabel(des)
    plt.ylabel(var)
    # Calculate the line of best fit
    coefficients = np.polyfit(t_filtered, y_filtered, 1)  # 1 indicates a linear fit
    poly = np.poly1d(coefficients)#(x)
    y_fit = poly(t_filtered)
    # Plot the line of best fit
    plt.plot(t_filtered, y_fit, color='red', linestyle='--')
    plt.title(var+" vs "+ttl)
    # Show the plot
    plt.show()
    
    # Example: enter your two arrays
    t_ = np.array(t_filtered)
    y_ = np.array(y_filtered)
    # Perform the significance test
    correlation_significance_test(t_, y_)


# In[46]:


# Hypotheses 5
#We also expect a positive correlation between time spent playing video games and performance in the Trail-Making tests (A & B)
#...as these activities both involve visual attention
# (and to Multiple Object Tracking) as video games often require this.
#We expect a larger magnitude of correlation in Trail-Making B than in Trail-Making A
#..as it additionally involves task switching (which many games involve).
#We believe gamers in general are used to “gamifying” tasks/reaching their peak potential and hence may perform better overall
#...and make a specific prediction that it will be correlated with better scores for Digit Symbol Matching.
                                      
df5 = df_recoded[["vgames","TMB Trail-Making (A)","TMB Trail-Making (B)","trail_making_avg",
                "TMB Multiple Object Tracking","TMB Digit Symbol Matching",
                "TMB Forward Digit Span","TMB Backward Digit Span","digit_span",
                  "TMB Paced Serial Addition"]]
df5_corr = df5.corr()
df5_corr.round(2)


# In[47]:


t = df5["vgames"]

for var in ["TMB Trail-Making (A)",	"TMB Trail-Making (B)", 'trail_making_avg',
            "TMB Multiple Object Tracking",	"TMB Digit Symbol Matching",
           "TMB Forward Digit Span","TMB Backward Digit Span","digit_span",
            "TMB Paced Serial Addition"]:
    print("Variable to compare to video game use is: ", var)
    
    y = df5[var]
    # Create a plot
    # Filter out NaN values from both x and y
    mask = ~np.isnan(t) & ~np.isnan(y) 
    t_filtered = t[mask]
    y_filtered = y[mask]
    
    plt.figure()
    plt.scatter(t_filtered, y_filtered, color='blue', marker='o')
    # Add labels and title
    plt.xlabel('video game usage')
    plt.ylabel(var)
    # Calculate the line of best fit
    coefficients = np.polyfit(t_filtered, y_filtered, 1)  # 1 indicates a linear fit
    poly = np.poly1d(coefficients)#(x)
    y_fit = poly(t_filtered)
    # Plot the line of best fit
    plt.plot(t_filtered, y_fit, color='red', linestyle='--')
    plt.title(var+" vs video game usage (hrs/day)")
    # Show the plot
    plt.show()
    
    # Example: enter your two arrays
    t_ = np.array(t_filtered)
    y_ = np.array(y_filtered)
    # Perform the significance test
    correlation_significance_test(t_, y_)


# In[48]:


# Compare means for binary measure (play video games: yes/no)

df5['vgames_any'] = df5['vgames']
df5['vgames_any'] = (df5['vgames'] > 0) *1.0
df5["vgames_no"]  = df5["vgames_any"]==0
df5["vgames_yes"] = df5["vgames_any"]==1

for var in ["TMB Trail-Making (A)",	"TMB Trail-Making (B)",	'trail_making_avg',
            "TMB Multiple Object Tracking",	"TMB Digit Symbol Matching",
           "TMB Forward Digit Span","TMB Backward Digit Span","digit_span",
            "TMB Paced Serial Addition"]:
    print(var)

    df5[var+"vgames_no"] = (df5[var])*(df5["vgames_no"]==1)
    df5[var+"vgames_yes"] = (df5[var])*(df5["vgames_yes"]==1)
    data_vgames_no = df5[var+"vgames_no"].values
    data_vgames_yes = df5[var+"vgames_yes"].values
    # Filter out 0s and NaNs
    filtered_data_vgames_no = data_vgames_no[(data_vgames_no != 0) & ~np.isnan(data_vgames_no)]
    filtered_data_vgames_yes = data_vgames_yes[(data_vgames_yes != 0) & ~np.isnan(data_vgames_yes)]


    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_vgames_no, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n2 = len(filtered_data_vgames_no)
    # Compute the standard error
    standard_error_vgames_no = std_dev / np.sqrt(n2)
    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_vgames_yes, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n1 = len(filtered_data_vgames_yes)
    # Compute the standard error
    standard_error_vgames_yes = std_dev / np.sqrt(n1)


    print("Without video games: ", "Mean: ", np.mean(filtered_data_vgames_no), "SE: ", standard_error_vgames_no)
    print("With video games: ", "Mean: ", np.mean(filtered_data_vgames_yes),"SE :", standard_error_vgames_yes)

    # Calculate sample variances
    s1_squared = np.var(filtered_data_vgames_no, ddof=1)
    s2_squared = np.var(filtered_data_vgames_yes, ddof=1)

    # Calculate degrees of freedom for Welch's t-test
    dof = ((s1_squared / n1) + (s2_squared / n2))**2 / \
         (((s1_squared / n1)**2 / (n1 - 1)) + ((s2_squared / n2)**2 / (n2 - 1)))


    # Perform independent two-sample t-test
    t_statistic, p_value = stats.ttest_ind(filtered_data_vgames_yes, filtered_data_vgames_no, equal_var=False)  # Set equal_var=False for Welch's t-test
    
    # Output the results
    print(f"T-statistic: {t_statistic}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p_value}")
    print("\n")


# In[49]:


# Hypotheses 6
# We expect that on the basis of research by (Mo et al., 2022) that individuals that
#...increase the speed when listening to lectures are better at self-regulated learning (Zhao et al., 2014) than those who do not.
# Consequently, we expect that students who report increasing the speed of lectures will have better results
# (than those who do not increase the speed) in the following tests
# as these tests involve elements amenable to improvement by self-regulated learning:
#1. Matrix Reasoning: The test was not time limited, allowing test-takers time to learn regularities in the rules
#...needed to solve the puzzle and note these (if needed).
#2. Digit Symbol Matching: Students may devise several on-line strategies to improve speed at this task
#...e.g., skipping the step of needing to match the symbol to the column (by merely memorising which symbol was associated with each digit)
#3. Visual Paired Associates Memory and
#4. Verbal Paired Associates Memory: Students could devise several strategies online to improve performance
#...such as imagining the pairs in similar locations in a room, having avatars speak the verbal paired associates and so on
#Digit Span (both 5. Forward Digit Span, and 6. Backward Digit Span)
# Students may also strategically employ ‘chunking’ to remember longer spans.


df6 = df_recoded[["TMB Matrix Reasoning", "TMB Digit Symbol Matching",
            "TMB Verbal Paired Associates Memory - Test","TMB Visual Paired Associates Memory - Test",
            "TMB Trail-Making (A)",	"TMB Trail-Making (B)",	"trail_making_avg",
            "TMB Multiple Object Tracking",	
             "TMB Forward Digit Span","TMB Backward Digit Span","digit_span",
                 "speed"]]

df6['speed'][(df_recoded['speed'] == "No")] = 0
df6['speed'][(df_recoded['speed'] == "Yes")] = 1
df6_corr = df6.corr()
df6_corr.round(2)


# In[50]:


for var in ["TMB Matrix Reasoning", "TMB Digit Symbol Matching",
            "TMB Verbal Paired Associates Memory - Test","TMB Visual Paired Associates Memory - Test",
            "TMB Trail-Making (A)",	"TMB Trail-Making (B)","trail_making_avg",
            "TMB Multiple Object Tracking",	
             "TMB Forward Digit Span","TMB Backward Digit Span","digit_span"]:


    df6[var+"speed_no"] = (df6[var])*(df6["speed"]==0)
    df6[var+"speed_yes"] = (df6[var])*(df6["speed"]==1)
    data_speed_no = df6[var+"speed_no"].values
    data_speed_yes = df6[var+"speed_yes"].values
    # Filter out 0s and NaNs
    filtered_data_speed_no = data_speed_no[(data_speed_no != 0) & ~np.isnan(data_speed_no)]
    filtered_data_speed_yes = data_speed_yes[(data_speed_yes != 0) & ~np.isnan(data_speed_yes)]

    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_speed_no, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n2 = len(filtered_data_speed_no)
    # Compute the standard error
    standard_error_speed_no = std_dev / np.sqrt(n2)
    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_speed_yes, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n1 = len(filtered_data_speed_yes)
    # Compute the standard error
    standard_error_speed_yes = std_dev / np.sqrt(n1)


    print("Normal speed ", "mean: ", np.mean(filtered_data_speed_no),"SE: ", standard_error_speed_no)
    print("Faster speed ", "mean: ", np.mean(filtered_data_speed_yes),"SE: ", standard_error_speed_yes)

    
    # Calculate sample variances
    s1_squared = np.var(filtered_data_speed_no, ddof=1)
    s2_squared = np.var(filtered_data_speed_yes, ddof=1)

    # Calculate degrees of freedom for Welch's t-test
    dof = ((s1_squared / n1) + (s2_squared / n2))**2 / \
         (((s1_squared / n1)**2 / (n1 - 1)) + ((s2_squared / n2)**2 / (n2 - 1)))


    # Perform independent two-sample t-test
    t_statistic, p_value = stats.ttest_ind(filtered_data_speed_no, filtered_data_speed_yes, equal_var=False)  # Set equal_var=False for Welch's t-test
    
    # Output the results
    print(f"T-statistic: {t_statistic}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p_value}")
    print("\n")


# In[51]:


#Hypotheses 7
#We believe that students who reported preparation before engaging in the tasks
#...(specifically, those who reported taking 10 seconds to steady their thoughts before beginning each test)
#...are more motivated to perform well in the tests and may be preparing their responses.
#We predict that this motivation and the possible engagement of strategic thinking will result in
#...those answering affirmatively to perform better in the following tests amenable to these qualities
#Visual Paired Associates Memory, Verbal Paired Associates Memory, Digit Span (both Forward Digit Span, and Backward Digit Span)
df7 = df_recoded[["preparation","TMB Verbal Paired Associates Memory - Test","TMB Visual Paired Associates Memory - Test",
         "TMB Forward Digit Span","TMB Backward Digit Span","digit_span"]]
df7['preparation'][(df_recoded['preparation'] == "No")] = 0
df7['preparation'][(df_recoded['preparation'] == "Yes")] = 1
df7_corr = df7.corr()
df7_corr.round(2)


# In[52]:


for var in ["TMB Verbal Paired Associates Memory - Test","TMB Visual Paired Associates Memory - Test",
         "TMB Forward Digit Span","TMB Backward Digit Span","digit_span"]:
    print(var)

    df7[var+"preparation_no"] = (df7[var])*(df7["preparation"]==0)
    df7[var+"preparation_yes"] = (df7[var])*(df7["preparation"]==1)
    data_preparation_no = df7[var+"preparation_no"].values
    data_preparation_yes = df7[var+"preparation_yes"].values
    # Filter out 0s and NaNs
    filtered_data_preparation_no = data_preparation_no[(data_preparation_no != 0) & ~np.isnan(data_preparation_no)]
    filtered_data_preparation_yes = data_preparation_yes[(data_preparation_yes != 0) & ~np.isnan(data_preparation_yes)]

    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_preparation_no, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n2 = len(filtered_data_preparation_no)
    # Compute the standard error
    standard_error_preparation_no = std_dev / np.sqrt(n2)
    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_preparation_yes, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n1 = len(filtered_data_preparation_yes)
    # Compute the standard error
    standard_error_preparation_yes = std_dev / np.sqrt(n1)


    print("No preparation ", "mean: ", np.mean(filtered_data_preparation_no),"SE: ", standard_error_preparation_no)
    print("With preparation ", "mean: ", np.mean(filtered_data_preparation_yes),"SE: ", standard_error_preparation_yes)

    
    # Calculate sample variances
    s1_squared = np.var(filtered_data_preparation_no, ddof=1)
    s2_squared = np.var(filtered_data_preparation_yes, ddof=1)

    # Calculate degrees of freedom for Welch's t-test
    dof = ((s1_squared / n1) + (s2_squared / n2))**2 / \
         (((s1_squared / n1)**2 / (n1 - 1)) + ((s2_squared / n2)**2 / (n2 - 1)))


    # Perform independent two-sample t-test
    t_statistic, p_value = stats.ttest_ind(filtered_data_preparation_no, filtered_data_preparation_yes, equal_var=False)  # Set equal_var=False for Welch's t-test
    
    # Output the results
    print(f"T-statistic: {t_statistic}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p_value}")
    print("\n")


# In[53]:


##### Hypotheses 8
# We predict that those that there will be positive correlations between self-reported use of two different emotion regulation techniques
# (1: use of journalling/deep breathing/conversation and 2: time spent listening to music) and the results in
# Multiracial Emotion Identification
# Multiracial Reading the Mind in the Eyes  
# This hypothesis is based on the simple assumption that those who are more in touch with their own emotions
#... and seek to regulate or express them more often will be able to better identify emotions in others.
# There is likely a negative correlation with age too (Gonçalves et al., 2018) and (Cortes et al., 2021)
df8 = copy.deepcopy(df_recoded)

# # Remove outliers
# df8.loc[1, "TMB Multiracial Reading the Mind in the Eyes"] = np.nan # 17, biggest outlier
# df8.loc[18, "TMB Multiracial Reading the Mind in the Eyes"] = np.nan # 17, biggest outlier
# df8.loc[25, "TMB Multiracial Reading the Mind in the Eyes"] = np.nan # 17, biggest outlier


# In[54]:


for IV in ["emotion_reg","music"]:
    
    for DV in ["TMB Multiracial Emotion Identification","TMB Multiracial Reading the Mind in the Eyes","multiracial"]:
    
        print("IV: ", IV)
        print("DV: ", DV)

        t = df8[IV].values
        y = df8[DV].values
        #t = [float(x) for x in t]
        #y = [float(x) for x in y]
        t_numeric = pd.to_numeric(t, errors='coerce')
        y_numeric = pd.to_numeric(y, errors='coerce')       
        # Create a plot
        # Filter out NaN values from both x and y
        mask = ~np.isnan(t_numeric) & ~np.isnan(y_numeric)
        t_filtered = t_numeric[mask]
        y_filtered = y_numeric[mask]
        
        plt.figure()
        plt.scatter(t_filtered, y_filtered, color='blue', marker='o')
        # Add labels and title
        plt.xlabel(IV)
        plt.ylabel(DV)
        # Calculate the line of best fit
        coefficients = np.polyfit(t_filtered, y_filtered, 1)  # 1 indicates a linear fit
        poly = np.poly1d(coefficients)
        y_fit = poly(t_filtered)
        # Plot the line of best fit
        plt.plot(t_filtered, y_fit, color='red', linestyle='--')
        plt.title(DV+" vs "+str(IV))
        # Show the plot
        plt.show()
        
        # Example: enter your two arrays
        t_ = np.array(t_filtered)
        y_ = np.array(y_filtered)
        # Perform the significance test
        correlation_significance_test(t_, y_)


# In[55]:


for var in ["TMB Matrix Reasoning", "TMB Digit Symbol Matching",
            "TMB Verbal Paired Associates Memory - Test","TMB Visual Paired Associates Memory - Test",
            "TMB Trail-Making (A)",	"TMB Trail-Making (B)","trail_making_avg",
            "TMB Multiple Object Tracking",	
             "TMB Forward Digit Span","TMB Backward Digit Span","digit_span"]:
    print(var)

    df8[var+"music_aesthetic"] = (df8[var])*(df8["music_type"]==0)
    df8[var+"music_thematic"] = (df8[var])*(df8["music_type"]==1)
    data_aesthetic = df8[var+"music_aesthetic"].values
    data_thematic = df8[var+"music_thematic"].values
    # Filter out 0s and NaNs
    filtered_data_aesthetic = data_aesthetic[(data_aesthetic != 0) & ~np.isnan(data_aesthetic)]
    filtered_data_thematic = data_thematic[(data_thematic != 0) & ~np.isnan(data_thematic)]

    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_aesthetic, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n1 = len(filtered_data_aesthetic)
    # Compute the standard error
    standard_error_aesthetic = std_dev / np.sqrt(n1)
    # Compute the standard deviation (sample)
    std_dev = np.std(filtered_data_thematic, ddof=1)  # ddof=1 for sample standard deviation
    # Compute the sample size
    n2 = len(filtered_data_thematic)
    # Compute the standard error
    standard_error_thematic = std_dev / np.sqrt(n2)


    print("Aesthetic music taste ", "mean: ", np.mean(filtered_data_aesthetic),"SE: ", standard_error_aesthetic)
    print("Thematic music taste ", "mean: ", np.mean(filtered_data_thematic),"SE: ", standard_error_thematic)

    
    # Calculate sample variances
    s1_squared = np.var(filtered_data_aesthetic, ddof=1)
    s2_squared = np.var(filtered_data_thematic, ddof=1)

    # Calculate degrees of freedom for Welch's t-test
    dof = ((s1_squared / n1) + (s2_squared / n2))**2 / \
         (((s1_squared / n1)**2 / (n1 - 1)) + ((s2_squared / n2)**2 / (n2 - 1)))


    # Perform independent two-sample t-test
    t_statistic, p_value = stats.ttest_ind(filtered_data_aesthetic, filtered_data_thematic, equal_var=False)  # Set equal_var=False for Welch's t-test
    
    # Output the results
    print(f"T-statistic: {t_statistic}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p_value}")
    print("\n")


# In[79]:


df_recoded.columns.values


# In[88]:


df_recoded['reading']


# In[86]:


# Import necessary libraries
import numpy as np
import statsmodels
import statsmodels.api as sm

cols_of_interest = [
                    #"age",
                    "sleep_hours", # "sleep_quality", # "sleep_overall", # # 
    
                    "tiktok_time", #"tiktok", "tiktok_cumulative", 
                    "phone_time",
    
                    "vgames", "chess_graded",
                    "music",
                    #"music_type",
                    # "music_num_genres",
                    # "music_play_instr",                    
                    # "languages_num",                    
                    "reading",
    
                    "stress",
                    #"mood", # mood not drawn independently from thepopulation
                    #"emotion_reg", #"emotion_reg_combo"#, "preparation", # "music",
                    # "exercise_hours", #"exercise_rating",
    
                    # "speed",
                    # "new_experiences",
                    # "creative"
                    
                   ]


#Multiple regression
arrays = [
            np.array(df_recoded[x]) for x in cols_of_interest
            ]

#Elements excluded due to the presence of nans
r = 0
nan_indices = []
for item in arrays:
    print(cols_of_interest[r])
    m = 0
    for elem in arrays[r]:
        if np.isnan(elem):
            if m not in nan_indices:
                nan_indices.append(m)
        m+=1
    r+=1
print(nan_indices)


#x = df_recoded['iq']
#x = df_recoded['iq_equalweight']
#x = df_recoded['iq_tmb_equalweight']
x = df_recoded['iq_normalised'] # This makes most sense, is what WAIS uses
X = np.column_stack(arrays)

# Create a mask to identify rows where all values are non-NaN
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(x)

# Apply the mask to filter out rows with NaNs
X_clean = X[mask]
x_clean = x[mask]

# Add a constant (intercept term) to the cleaned independent variables
X_clean = sm.add_constant(X_clean)

# Create the OLS model using the cleaned data
model = sm.OLS(x_clean, X_clean)

# Fit the model
results = model.fit()


# Print the summary, which includes statistical significance
print(results.summary())


# In[70]:


## BElow out of interest not in project


# In[105]:


#### Find any pair with statistically significant relationship
max_p = 0.15

correlation_matrix = df_recoded.corr()

# Create an empty list to store pairs of columns with correlation > x
significant_pairs = {}
pairs_tried = []
errors = []

# Iterate through the correlation matrix to find pairs with correlation > x
for col1 in correlation_matrix.columns:
    for col2 in correlation_matrix.columns:
        if col1!=col2 and (col1, col2) not in pairs_tried and (col2, col1) not in pairs_tried:
            try:
                r, p = correlation_significance_test(df_recoded[col1].values,df_recoded[col2].values)
                if p <= max_p:
                    significant_pairs[(col1, col2)] = (round(r,3), round(p,3))
                pairs_tried.append((col1,col2))
            except:
                errors.append((col1,col2))
                print("Error", col1, " : ", col2)

from IPython.display import clear_output
# Clear the output
clear_output(wait=False)


# In[106]:


# Display all the pairs with significant relationships
for key in significant_pairs.keys():
    print(key,significant_pairs[key])


# In[107]:


# Count the variables which have most relationships to others
elems_a = [elem[0] for elem in significant_pairs]
elems_b = [elem[1] for elem in significant_pairs]

num_correlated_to_a = Counter(elems_a)
num_correlated_to_b = Counter(elems_b)
num_correlated_to = num_correlated_to_a + num_correlated_to_b
print(num_correlated_to)


# In[108]:


# Find all significantly correlated with x
find_var = "TMB Cambridge Face Memory Test"
plot_all(find_var)


# In[109]:


# Find all significantly correlated with x
find_var = "TMB Multiracial Emotion Identification"
plot_all(find_var)


# In[111]:


# Find all significantly correlated with x
find_var = "music_type"
plot_all(find_var)


# In[110]:


# Find all significantly correlated with x
find_var = "music"
plot_all(find_var)


# In[102]:


# Find all significantly correlated with x
find_var = "multiracial"
plot_all(find_var)


# In[63]:


#### Find any pair with correlation > x or < - x
x = 0.4
correlation_matrix = df_recoded.corr()
pairs = []

# Create an empty list to store pairs of columns with correlation > x
high_correlation_pairs = []

# Iterate through the correlation matrix to find pairs with correlation > x
for col1 in correlation_matrix.columns:
    for col2 in correlation_matrix.columns:
        corr = correlation_matrix.loc[col1, col2]
        if col1 != col2 and abs(corr) > x:
            pair = [col1, col2, corr]
            pair_rev = [col2, col1, corr]

            #print(pair,
            # Check if the pair (or its reverse) already exists
            if pair not in high_correlation_pairs and pair_rev not in high_correlation_pairs:
                high_correlation_pairs.append(pair)


# In[64]:


# Display high correlations between TMB variables
ranked_pairs = sorted(high_correlation_pairs, key=lambda x: abs(x[2]), reverse=True)
for pair in ranked_pairs:
    if "TMB" in pair[0] and "TMB" in pair[1] and pair[0] not in pair[1] and pair[1] not in pair[0]:
        print(f"Correlation: {pair[2]:.2f} {pair[0]} and {pair[1]}")


# In[65]:


# Display high correlations between TMB and non-TMB
ranked_pairs = sorted(high_correlation_pairs, key=lambda x: abs(x[2]), reverse=True)
for pair in ranked_pairs:
    if ("TMB" not in pair[0] and "TMB" in pair[1]) or ("TMB" in pair[0] and "TMB" not in pair[1]):
        print(f"Correlation: {pair[2]:.2f} {pair[0]} and {pair[1]}")


# In[66]:


# Display high correlations between non-TMB
ranked_pairs = sorted(high_correlation_pairs, key=lambda x: abs(x[2]), reverse=True)
for pair in ranked_pairs:
    if "TMB" not in pair[0] and "TMB" not in pair[1]:
        print(f"Correlation: {pair[2]:.2f} {pair[0]} and {pair[1]}")


# In[67]:


df_all_tmb = df_recoded[[col for col in df_recoded.columns if "TMB" in col]]
df_all_tmb_corr = df_all_tmb.corr()
df_all_tmb_corr.round(2)

