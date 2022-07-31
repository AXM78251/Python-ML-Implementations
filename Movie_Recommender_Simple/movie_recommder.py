# Movie recommender system implemented in Python
# Very basic model, will attempt to recommend similar movies
# Relies on content-based filtering
# Unsupervised learning, will rely on the IMDB dataset containing movies from 2006 to 2016
# Dataset citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872

# Import the necessary python modules and libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read our movies csv file and turn it into a dataframe
# Contains movie metadata from IMDB
IMDB_MOVIES = "IMDB-Movie-Data.csv"
movies = pd.read_csv(IMDB_MOVIES, header=0, dtype=str)
movies['Movie_ID'] = range(0, 1000)

# This list will contain the list of all possible movies from our dataset
movie_titles = []

# This variable will keep track of how many times we have run the program
times_ran = 0

# This list will contain different words that all mean yes
yes_words = ["yes", 'sure', "all right", "okay", "yeah", "yea", "ok", "y"]


def get_relevant_data():
    """
    This function will extract relevant data from our IMDB movies dataset and add them
    as a new feature column to our movies dataframe

    This function will also append all movie titles to our movie_titles list

    :return: returns a list of the most relevant metadata pertaining to our movies dataset

    """
    rev_data = []
    for i in range(1000):
        rev_data.append(movies['Title'][i] + ' ' + movies['Genre'][i] + ' ' + movies['Director'][i]
                        + ' ' + movies['Actors'][i] + ' ' + movies['Rating'][i])
        movie_titles.append(movies['Title'][i].lower())
    return rev_data


# Now create a new column in our original data which will contain all our relevant features
movies['Important Features'] = get_relevant_data()

# Create our matrix of token counts using our important features column from our movies dataframe
matrix = CountVectorizer().fit_transform(movies['Important Features'])

# Get the cosine similarity of our token counts matrix
cos_sim = cosine_similarity(matrix)


def get_movie_id(movie_name, num_recs):
    """
    This function will take into two arguments and get the movie id of our favorite movie

    This function will also call our sort_similar_scores function

    If movie name is not valid or is not found, it will ask users for a different movie title input

    :param movie_name: name of our favorite movie
    :param num_recs: number of recommendations we are looking
    :return: nothing, calls pur get_similar_scores function

    """
    name = movie_name.lower()
    exists = movie_titles.count(name)
    if exists == 0:
        print("\n" + "Movie not found, please try a different movie" + "\n")
        run_program(False)
    else:
        movie_id = movie_titles.index(name)
        sort_similar_scores(movie_id, num_recs)


def sort_similar_scores(movie_id, num_recs):
    """

    This function will take in two arguments and sort our cosine similarity data in decreasing order

    If an invalid movie id is passed in, it will ask user to input a different movie title into our program

    At the end, this function will call our get_sim_movies function

    :param movie_id: the id of the movie title we passed in at the beginning of our program
    :param num_recs: the number of recommendations we are looking for
    :return: nothing, calls our get_sim_movies function

    """
    if movie_id < 0 or movie_id > 999:
        print("\n" + "Invalid movie index, please try again" + "\n")
    else:
        similarity_scores = list(enumerate(cos_sim[movie_id]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:]
    get_sim_movies(similarity_scores, movies[movies.Movie_ID == movie_id].Title.values[0], num_recs)


def get_sim_movies(sim_scores, movie, num_recs):
    """
    This function takes in 3 arguments and will generate a list of the most similar movies to our favorite movie
    The number of most similar movies will be determined by the number of recommendations we are looking for

    This function will also call our _sim_movies function

    :param sim_scores: list of data sorted in descending error depicting the
                       similarity between the movie passed in and every other movie
    :param movie: the name of our favorite movie
    :param num_recs: the number of recommendations we are looking for
    :return: nothing, calls our _sim_movies function

    """
    similar_movies = []
    for i in range(num_recs):
        movie_id = sim_scores[i][0]
        movie_title = find_movie(movie_id)
        similar_movies.append(movie_title)
    _sim_movies(movie, similar_movies, num_recs)


def find_movie(movie_id):
    """

    This function takes in 1 argument and will identify the name of the movie based on the movie id passed in

    :param movie_id: the id of the movie we are trying to find
    :return: returns the name of the movie corresponding to the movie id passed in

    """
    return movies[movies.Movie_ID == movie_id].Title.values[0]


def _sim_movies(movie, sim_movies, num_recs):
    """

    This function takes in 3 arguments and will print out the most similar movies to our favorite movie
    The number of most similar movies will be determined by the number of recommendations we are looking for

    This function will also call our run_program to restart our program to find more recommendations
    Whether that is for the same movie or a different movie, only if the user wants more recommendations

    :param movie: the name of our favorite movie
    :param sim_movies: a list containing the n most similar movies to our favorite movie
                       n is dictated by the number of recommendations we are looking for
    :param num_recs: the number of recommendations we are looking for
    :return: nothing, will call function to start program over if we want more recommendations

    """
    print("\n")
    phrase1 = "If you liked: "
    # phrase1 = phrase1.center(15)
    print(phrase1 + movie)
    print("Then you might consider watching..." + "\n")
    for i in range(num_recs):
        print(" " + str(i + 1) + ". " + str(sim_movies[i]))
    print("\n")
    phrase3 = input("Would you like recommendations for other movies? ")
    phrase3 = phrase3.strip().lower()
    if phrase3 in yes_words:
        run_program(False);
    else:
        print("Thanks for using this movie recommendation system. Have a great day!")
        exit(0)


def run_program(first_time):
    """

    The start of our program

    This function will ask the user for their favorite movie
    As well as the number of recommendations they are looking for

    This function will also call our _get_movie_id function and execute the rest of our program

    :param first_time: variable that allows us to determine whether it is our first time running the program
    :return: nothing, this function will call get_movie_id which will execute the rest of our progran

    """
    if first_time:
        print("\n" + "Welcome to the movie recommendation system! Glad to see you here!")
    else:
        print("Welcome back!")
    fav_movie = input("Please enter your favorite movie from 2006 - 2016: ")
    fav_movie = fav_movie.strip()
    num_recs = input("How many recommendations would you like? ")
    num_recs = int(num_recs)
    get_movie_id(fav_movie, num_recs)


# Begin our program
run_program(True)
