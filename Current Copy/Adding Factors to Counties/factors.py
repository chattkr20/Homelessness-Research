import pandas as pd
import numpy as np
import matplotlib as plot
from pandas import DataFrame

def filtered(county):
        filtered_county = county
        filtered_county = filtered_county.replace("'", "")
        filtered_county = filtered_county.replace("Ft", "")
        filtered_county = filtered_county.replace(" and ", "")
        filtered_county = filtered_county.replace(" ", "")
        filtered_county = filtered_county.replace("City&County", "")
        filtered_county = filtered_county.replace("City", "")
        filtered_county = filtered_county.replace("CoC", "")
        filtered_county = filtered_county.replace("County", "")
        filtered_county = filtered_county.replace("Counties", "")
        filtered_county = filtered_county.replace("CensusArea", "")
        filtered_county = filtered_county.replace("Municipality", "")
        filtered_county = filtered_county.replace("Borough", "")
        filtered_county = filtered_county.replace("Beach", "")
        filtered_county = filtered_county.replace("North", "")
        filtered_county = filtered_county.replace("South", "")
        filtered_county = filtered_county.replace("East", "")
        filtered_county = filtered_county.replace("West", "")
        filtered_county = filtered_county.replace("east", "")
        filtered_county = filtered_county.replace("west", "")
        filtered_county = filtered_county.replace("Valley", "")
        filtered_county = filtered_county.replace("Peninsula", "")
        filtered_county = filtered_county.replace("Area", "")
        filtered_county = filtered_county.replace("Regional", "")
        filtered_county = filtered_county.replace("Hills", "")
        filtered_county = filtered_county.replace("Upper", "")
        filtered_county = filtered_county.replace("Lower", "")
        filtered_county = filtered_county.replace("Township", "")
        filtered_county = filtered_county.replace("Town", "")
        filtered_county = filtered_county.lower()

        return filtered_county

state_names = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

df = pd.read_csv("Adding Factors to Counties/counties.csv")
county_array = list(df.to_numpy())

homeless = pd.read_csv("Data/homelessness.csv")
homeless.drop(homeless.columns.difference(['CoC Number', 'CoC Name', 'Overall Homeless, 2022']), 1, inplace=True)
homeless = list(homeless.to_numpy())

rent = pd.read_csv("Data/rent.csv")
rent.drop(rent.columns.difference(['RegionName', 'RegionType', 'State', '2021-05-31']), 1, inplace=True)
rent = list(rent.to_numpy())

poverty = pd.read_csv("Data/poverty.csv")
poverty.drop(poverty.columns.difference(['Stabr', 'Area_name','Attribute', 'Value']), 1, inplace=True)
poverty = list(poverty.to_numpy())

unemployment = pd.read_csv("Data/unemployment.csv")
unemployment.drop(unemployment.columns.difference(['State','Area_Name','Attribute','Value']), 1, inplace=True)
unemployment = list(unemployment.to_numpy())

edu_rate = pd.read_csv("Data/education_rate.csv", encoding='latin-1')
edu_rate.drop(edu_rate.columns.difference(['State','Area name','Attribute','Value']), 1, inplace=True)
edu_rate = list(edu_rate.to_numpy())



temp = []
for row in rent:
        if row[1] == "county":
                temp.append(list(row))
rent = temp

temp = []
for row in poverty:
        if (not row[2].find('POVALL') < 0) and not (row[1] == "United States" or row[0] in state_names):
                temp.append(list(row))
poverty = temp

temp = []
for row in unemployment:
        if (not (row[1] == "United States" or row[1] in state_names)) and (row[2] == "Unemployed_2021" or row[2] == "Unemployment_rate_2021"):
                temp_row = row
                temp_row[1] = temp_row[1][0:temp_row[1].find(",")]
                temp.append(list(temp_row))
unemployment = temp

temp = []
for row in edu_rate:
        if (not (row[1] == "United States" or row[1] in state_names)) and (row[2] == "Less than a high school diploma, 2017-21" or row[2] == "High school diploma only, 2017-21"):
                temp.append(list(row))
edu_rate = temp



count1 = 0
count2 = 0
for i in range(len(county_array)-1, -1, -1):

        county_array[i] = list(county_array[i])

        worked = False
        for row in homeless:
                if county_array[i][0] == row[0]:
                        county_array[i].append(str(row[2].replace(',', '')))
                        worked = True
        if not worked:
                county_array[i].append("none")


        worked = False
        for row in rent:
                row = list(row)
                if filtered(str(county_array[i][3])) == filtered(str(row[0])) and (row[2] == county_array[i][0][0:2]):
                        county_array[i].append(str(row[3]))
                        worked = True
        if not worked:
                county_array[i].append("none")


        worked = False
        for row in poverty:
                if filtered(str(county_array[i][3])) == filtered(str(row[1])) and (row[0] == county_array[i][0][0:2]):
                        county_array[i].append(int(row[3]))
                        worked = True
        if not worked:
                county_array[i].append("none")
                county_array[i].append("none")


        worked = False
        for row in unemployment:
                if filtered(str(county_array[i][3])) == filtered(str(row[1])) and (row[0] == county_array[i][0][0:2]):
                        county_array[i].append(int(row[3]))
                        worked = True
        if not worked:
                county_array[i].append("none")
                county_array[i].append("none")


        worked = False
        for row in edu_rate:
                if filtered(str(county_array[i][3])) == filtered(str(row[1])) and (row[0] == county_array[i][0][0:2]):
                        county_array[i].append(int(row[3]))
                        worked = True
        if not worked:
                county_array[i].append("none")
                county_array[i].append("none")

        
        county_array[i] = np.array(county_array[i])
        if i % 100 == 0:
                print(i)


df = DataFrame(county_array)
df.to_csv("Adding Factors to Counties/Output.csv")