# This is an anime recommendation system
# Utilizes an item based collaborative filtering system
# Will rely on data from a dataset found on kaggle.com
# Link to dataset: https://www.kaggle.com/CooperUnion/anime-recommendations-database

# Import the necessary python modules and libraries
import random

import numpy as np
import pandas as pd

""" These next several lines of code will set up our data for the execution of the rest of the program """

# Read data from the anime_ratings.dat file and turn it into a dataframe
ANIME_RATINGS = 'anime/anime_ratings.dat'
ratings = pd.read_csv(ANIME_RATINGS, delimiter="\t")

# Rename the columns of our dataframe
ratings.columns = ['user_ids', 'anime_ids', 'rating']

# Read the metadata from the anime_info.data and turn it into a dataframe
ANIME_INFO = 'anime/anime_info.dat'
info = pd.read_csv(ANIME_INFO, delimiter="\t")

# Now extract only the relevant columns of our data which will include the anime name and its corresponding id number
info = info[['anime_ids', 'name']]

# Now combine our data to be more easily analyzed
combined_data = pd.merge(ratings, info, on='anime_ids')

# To make our data a bit smaller and easier to compute, we will perform some explanatory data analysis (EDA)
# We will cut down our data to only anime who have more than 100 ratings
# First group anime by name
all_anime = combined_data.groupby('name')

# Now add anime one by one with a new value which will be the number of ratings for that given anime
# Make sure to reset our indices
all_anime = all_anime.agg(number_of_ratings=('rating', 'count')).reset_index()

# Now count down our data to only the anime with more than 100 ratings
all_anime = all_anime[all_anime['number_of_ratings'] > 100]

# Now perform another merge once again since we know have the anime with more than 100 ratings
combined_data = pd.merge(combined_data, all_anime['name'], on='name', how='inner')

# Now create a spreadsheet style pivot table where the values are our ratings
# The user_ids are our columns and the anime title are our rows
spreadsheet = combined_data.pivot_table(values='rating', index='name', columns='user_ids')

# Now subtract the user mean across all anime the user has watched
# Each column vector represents all the anime any given user has watched so we want to take the column mean
spreadsheet = spreadsheet.sub(spreadsheet.mean(axis=1), axis=0)

# Now we want to get the item similarity matrix using pearson correlation
# Be sure to transpose our spreadsheet style data first
anime_similarity = spreadsheet.T.corr()


def predict_anime_rating(anime_name, watched_anime):
    """
    This function will attempt to predict the rating the user will give an unwatched title
    By using a list of anime the user has watched and rated

    :param anime_name: The name of the unwatched anime title whose rating this system will attempt to predict
    :param watched_anime: A list containing the anime that the user has watched and rated
    :return: A tuple containing the name of the unwatched anime along with it's predicted rating

    """
    # Start by getting the similarity between the given anime and every other anime
    similarity = anime_similarity[[anime_name]]

    # Reset our indices and rename our column to something more reader friendly
    similarity = similarity.reset_index().rename(columns={anime_name: 'similarity_scores'})

    # Now merge together our similarity scores and the list of anime the user has watched and rated
    # Merges them at the column that contains all the titles of the anime
    merge = pd.merge(watched_anime, similarity, on='name', how='inner')

    # Now sort out the data based on the similarity scores
    merge = merge.sort_values(by='similarity_scores', ascending=False)

    # Now get only the top 5 most similar anime as these will be what we use to make our predicted rating
    most_similar = merge[:5]

    # Now calculate our predicted rating
    # We want to get the weighted rating where the anime with higher similarity get more weight
    predicted_rating = round(np.average(most_similar['rating'], weights=most_similar['similarity_scores']), 5)

    # Now return the name of the unwatched anime along with its predicted rating
    return anime_name, predicted_rating


def predictions(user_id, n):
    """
    This function will calculate the predicted ratings for all the anime the specified user has not watched
    It will then provide recommendations for the user based on the n anime with the highest predicted scores

    :param user_id: The id number of the user who we want to make predictions for
    :return: Nothing, just prints out the top n anime the user has watched and the best n recommendations for the user

    """
    # First attain the list of all the anime the user has watched and rated
    # Remember that our column vectors represent all the anime a designated user has watched
    # Also make sure to drop all the anime that the user has not watched
    watched_anime = spreadsheet[user_id].dropna(axis=0, how='all')

    # Now sort our data using the ratings in descending fashion
    watched_anime = watched_anime.sort_values(ascending=False)

    # Now reset our indices and rename our column to something more user-friendly
    watched_anime = watched_anime.reset_index().rename(columns={user_id: 'rating'})

    # Now begin creating the list of anime that the user has not watched
    unwatched_anime = []

    # This list will contain all the anime titles featured in our dataset
    all_anime = spreadsheet.index.values

    for title in all_anime:
        if title not in watched_anime['name']:
            unwatched_anime.append(title)

    # Now go ahead and find the predicted rating for every anime that the user has not watched
    all_ratings = [predict_anime_rating(x, watched_anime) for x in unwatched_anime]

    # Now sort our ratings in descending order
    sorted_ratings = sorted(all_ratings, key=lambda x: x[1])[::-1]

    # Now we can go ahead and print out the n favorite anime the user has watched
    # Along with the top n recommendations this system has attempted to find
    # If the user has watched less tha n anime just print out those n anime in descending order by rating
    num_favorites = n
    if (n > len(watched_anime['name'])):
        num_favorites = len(watched_anime['name'])

    print("\n" + "The top " + str(num_favorites) + " anime that user " + str(user_id) + " has watched include:" + "\n")

    for i in range(num_favorites):
        print(str(i + 1) + ". " + watched_anime['name'][i])

    print("\n" + "Based on user " + str(user_id) + "'s favorite anime, some good recommendations include: " + "\n")

    for i in range(num_favorites):
        print(str(i + 1) + ". " + sorted_ratings[i][0])


def predict_random_user():
    """
    This function will select a random user to find recommendations, will call our other functions
    To find the desired recommendations

    :return: A call to our start program to (possibly) get recommendations for another random user

    """
    all_users = spreadsheet.columns.values
    random_user = random.choice(all_users)

    random_num_recs = random.choice([5, 6, 7, 8, 9, 10])
    predictions(random_user, random_num_recs)

    # Now go back to start of program to get predictions for another random user from our dataset
    return start_program(False)


def predict_my_mal(num_recs):
    """
    This function will print out good recommendations for me using my mal list

    :return: Nothing, will just print out my top n favorite along with the best n anime recommendations

    """
    # Read our data and turn it into a dataframe
    my_list = pd.read_csv("my_mal.csv")

    # Get only the anime that I have completed
    my_list = my_list[my_list['my_status'] == 'Completed']

    # Keep only the relevant columns of metadata
    my_list = my_list[['series_title', 'my_score']]

    # Rename our columns
    my_list.columns = ['name', 'rating']

    # Find and subtract the mean from all the anime I have watched
    my_list['rating'] = my_list['rating'] - np.mean(my_list['rating'])

    # Now, sort my anime in descending order using the ratings as a guide
    my_list = my_list.sort_values(by='rating', ascending=False)

    # Now create a list of all the anime I have not watched
    unwatched_anime = []
    for title in spreadsheet.index.values:
        if title not in my_list['name'].values:
            unwatched_anime.append(title)

    # Now predict the ratings of all my unwatched anime
    all_ratings = [predict_anime_rating(x, my_list) for x in unwatched_anime]

    # Sort all our ratings in descending order
    sorted_ratings = sorted(all_ratings, key=lambda x: x[1])[::-1]

    recs = num_recs
    if (recs > len(my_list['name'].values)):
        recs = len(my_list['name'].values)

    # Now print out my n favorite anime where n is dictated by a random number between 5 and 10
    print("\n" + "My top " + str(recs) + " anime are:" + "\n")
    for i in range(recs):
        print(str(i + 1) + ". " + my_list['name'].values[i])

    # Now print out the top n recommendations that this system has found for me
    print("\n" + "Based on these anime, this system finds some suitable recommendations to be: " + "\n")
    for i in range(recs):
        print(str(i + 1) + ". " + sorted_ratings[i][0])

    # Now go back to start of program to get predictions for a random user from our dataset
    start_program(False)


def start_program(first_time):
    """
    This function will be the start of our program, will find recommendations for a random user
    Or for myself based on user input

    :param first_time: Variable that determines whether to find recommendations for us or a random user
    :return: Nothing, just executes rest of program

    """
    # This list will contain different words that all mean yes
    yes_words = ["yes", 'sure', "all right", "okay", "yeah", "yea", "ok", "y"]

    if first_time:
        print("Welcome to this anime recommendation system! " + "\n")
        num_recs = input("How many recommendations would you like to see for yourself? ")
        print("\n" + "Great! This system will now attempt to find " + str(num_recs) + " recommendations!")
        predict_my_mal(int(num_recs))
    else:
        print("\n" + "Welcome back!!!" + "\n")
        continue_program = input("Would you like to find recommendations for a random user now? ")
        if continue_program in yes_words:
            print("\n" + "Great! This system will now find recommendations for a random user!")
            predict_random_user()
        else:
            print("\n" + "Okay. Thanks for using this recommendation system! Have a great day! ")


start_program(True)
