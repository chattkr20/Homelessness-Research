import pandas as pd
import numpy as np
import matplotlib as plot
from pandas import DataFrame

state_names = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

# df = pd.read_csv("Data/CoCtoCounty.csv")

# myList = []
# for CoCName in df.to_numpy():
#         counties = CoCName[1].split("+")
#         temp = []
#         for county in counties:
#                 if county.find("/") >= 0:
#                         splitCounty = county.split("/")
#                         for newCounty in splitCounty:
#                                 temp.append(newCounty)
#                 else:
#                         temp.append(county)
#         counties = temp
#         myList.append(counties)

# df = df.drop(df.columns[[1]], axis=1, inplace=False)
# df = pd.concat([df, DataFrame(myList)], axis=1)

# df.to_csv("Separating Counties/CoC_Output.csv")
# print(df)

############################################################

# ruleOuts = pd.read_csv("Separating Counties/ruleOut.csv")
# ruleOuts = np.array(ruleOuts)
# temp = []
# for row in ruleOuts:
#         temp.append(row[2])
# ruleOuts = temp

def filtered(county):
        filtered_county = county
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

# df = pd.read_csv("Separating Counties/CoC Info - CoC_Output.csv")
# array = df.to_numpy()
# temp = []
# for row in array:
#         for i in range(2, len(row)):
#                 if pd.isna(row[i]):
#                         pass
#                 else:
#                         if not (row[i] in ruleOuts):
#                                 temp.append([row[0], row[1], filtered(row[i])])

# array = temp
# df = DataFrame(array)
# df.to_csv("Separating Counties/CoC_Output.csv")

############################################################
#https://simplemaps.com/data/us-cities


# counties = pd.read_csv("Separating Counties/USA_Counties.csv")
# counties.drop(counties.columns.difference(['NAME','STATE_NAME']), 1, inplace=True)
# county_array = np.array(counties)
# tempCounties = county_array
# for i in range(len(county_array)):
#         county_array[i][0] = filtered(county_array[i][0])

# cities = pd.read_csv("Separating Counties/uscities.csv")
# cities.drop(cities.columns.difference(['city','state_name', 'county_name']), 1, inplace=True)
# cities_array = np.array(cities)
# tempCities = cities_array
# for i in range(len(cities_array)):
#         cities_array[i][0] = filtered(cities_array[i][0])
#         cities_array[i][2] = filtered(cities_array[i][2])

# df = pd.read_csv("Separating Counties/all_sections_without_directions.csv")
# cocArray = np.array(df)
# tempCoC = cocArray
# for i in range(len(cocArray)):
#         cocArray[i][3] = filtered(cocArray[i][3])

# count = 0
# finalArray = []
# for i in range(len(cocArray)-1, -1, -1):
#         match = False
#         for j in range(len(county_array)-1, -1, -1):
#                 if county_array[j][0] == cocArray[i][3] and county_array[j][1] == cocArray[i][2] and not match:
#                         finalArray.append([tempCoC[i][1], tempCoC[i][2], tempCoC[i][3], tempCounties[j][0]])
#                         match = True
#         if not match:
#                 for j in range(len(tempCities)-1, -1, -1):
#                         if cities_array[j][0] == cocArray[i][3] and cities_array[j][1] == cocArray[i][2] and not match:
#                                 finalArray.append([tempCoC[i][1], tempCoC[i][2], tempCoC[i][3], tempCities[j][2]])
#                                 match = True
#         if not match and (cocArray[i][3].find("balanceofstate") < 0 and cocArray[i][3].find("statewide") < 0):
#                 match = True
#         if not match:
#                 finalArray.append([tempCoC[i][1], tempCoC[i][2], tempCoC[i][3], "BAL. OF STATE/STATEWIDE"])
#                 match = True

# df = DataFrame(finalArray)
# df.to_csv("Separating Counties/CoC_Output.csv")


############################################################


# count = 0
# count2 = 0
# df = pd.read_csv("Separating Counties/state-CoC-County.csv")
# array = df.to_numpy()
# array = list(array)

# temp = []

# for i in range(len(county_array)-1, -1, -1):
#         for j in range(len(array)):
#                 if array[j][3] == county_array[i][0] and array[j][1] == county_array[i][1]:
#                         county_array = np.delete(county_array, i, 0)

# for i in range(len(array)-1, -1, -1):
#         if array[i][3] == "BAL. OF STATE/STATEWIDE":
#                 temp.append([array[i][0], array[i][1], array[i][2]])

# for county in county_array:
#         for row in temp:
#                 if row[1] == county[1]:
#                         array.append([row[0], row[1], row[2], county[0]])
#                         print([row[0], row[1], row[2], county[0]])

# array = np.asarray(array)
# df = DataFrame(array)
# df.to_csv("Separating Counties/CoC_Output.csv")