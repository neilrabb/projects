
from utils import mean

#############################
# Phase 1: Data Abstraction #
#############################


# Reviews

def make_review(restaurant_name, rating):
    """Return a review data abstraction."""
    return [restaurant_name, rating]

def review_restaurant_name(review):
    """Return the restaurant name of the review, which is a string."""
    return review[0]

def review_rating(review):
    """Return the number of stars given by the review, which is a
    floating point number between 1 and 5."""
    return review[1]


# Users

def make_user(name, reviews):
    """Return a user data abstraction."""
    return [name, {review_restaurant_name(r): r for r in reviews}]

def user_name(user):
    """Return the name of the user, which is a string."""
    return user[0]

def user_reviews(user):
    """Return a dictionary from restaurant names to reviews by the user."""
    return user[1]


### === +++ USER ABSTRACTION BARRIER +++ === ###

def user_reviewed_restaurants(user, restaurants):
    """Return the subset of restaurants reviewed by user.

    Arguments:
    user -- a user
    restaurants -- a list of restaurant data abstractions
    """
    names = list(user_reviews(user))
    return [r for r in restaurants if restaurant_name(r) in names]

def user_rating(user, restaurant_name):
    """Return the rating given for restaurant_name by user."""
    reviewed_by_user = user_reviews(user)
    user_review = reviewed_by_user[restaurant_name]
    return review_rating(user_review)


# Restaurants

def make_restaurant(name, location, categories, price, reviews):
    """Return a restaurant data abstraction containing the name, location,
    categories, price, and reviews for that restaurant."""
    # BEGIN Question 2
    return [name, location, categories, price, reviews]
    # END Question 2

def restaurant_name(restaurant):
    """Return the name of the restaurant, which is a string."""
    # BEGIN Question 2
    return restaurant[0]
    # END Question 2

def restaurant_location(restaurant):
    """Return the location of the restaurant, which is a list containing
    latitude and longitude."""
    # BEGIN Question 2
    return restaurant[1]
    # END Question 2

def restaurant_categories(restaurant):
    """Return the categories of the restaurant, which is a list of strings."""
    # BEGIN Question 2
    return restaurant[2]
    # END Question 2

def restaurant_price(restaurant):
    """Return the price of the restaurant, which is a number."""
    # BEGIN Question 2
    return restaurant[3]
    # END Question 2

def restaurant_ratings(restaurant):
    """Return a list of ratings, which are numbers from 1 to 5, of the
    restaurant based on the reviews of the restaurant."""
    return [review_rating(x) for x in restaurant[4]]
recommend.py
"""A Yelp-powered Restaurant Recommendation Program"""

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #
##################################


def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location.
    If multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    """
    # BEGIN Question 3
    return min(centroids,key = lambda x:distance(location,x))
mhao@berkeley.edu commented 1 year, 1 month ago
Good use of the min function! Next time, maybe try using a more descriptive variable name like centroid instead of x.

    # END Question 3


def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    # BEGIN Question 4
    restaurant_locations = [restaurant_location(x) for x in restaurants]
    groups = []
    k = 0
    for restaurant in restaurant_locations:
        groups.append([find_closest(restaurant,centroids),restaurants[k]])
        k += 1
    return group_by_first(groups)

def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""
    # BEGIN Question 5
    x_values = []
    y_values = []
    restaurant_locations = []
    for restaurant in cluster:
        restaurant_locations.append(restaurant_location(restaurant))
    x_values = [x[0] for x in restaurant_locations]
    y_values = [y[1] for y in restaurant_locations]
    return [mean(x_values),mean(y_values)]

    # for coordinate in restaurant_locations:
    #     x_values.append(restaurant_locations[0])
    #     print(x_values)
    #     y_values.append(restaurant_locations[1])
    #     print(y_values)


    # END Question 5


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0
    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
        # BEGIN Question 6
        clusters = group_by_centroid(restaurants,centroids)
        centroids = [find_centroid(cluster) for cluster in clusters]
        # END Question 6
        n += 1
    return centroids


################################
# Phase 3: Supervised Learning #
################################


def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    reviews_by_user = {review_restaurant_name(review): review_rating(review)
        for review in user_reviews(user).values()}

    xs = [feature_fn(r) for r in restaurants]
    ys = [reviews_by_user[restaurant_name(r)] for r in restaurants]

    # BEGIN Question 7
    sum_squares_y_y_list = [x - mean(ys) for x in ys]
    sum_squares_x_x_list = [x - mean(xs) for x in xs]
    sum_squares_both_list = zip(sum_squares_x_x_list, sum_squares_y_y_list)
    sum_squares_y_y = sum([pow(x, 2) for x in sum_squares_y_y_list])
    sum_squares_x_x = sum([pow(x, 2) for x in sum_squares_x_x_list])
    sum_squares_both_total = sum([x[0] * x[1] for x in sum_squares_both_list])
    b = sum_squares_both_total/ sum_squares_x_x
    a = mean(ys) - b * mean(xs)
    r_squared = pow(sum_squares_both_total, 2) / (sum_squares_x_x * sum_squares_y_y)  #
    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared
    # END Question 7


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 8
    return max([find_predictor(user, reviewed, x) for x in feature_fns], key = lambda y: y[1])[0]
    # END Question 8

    # xs is the extracted feature value for each restaurant in restaurants
    # ys is the user's ratings for the restaurants in restaurants

def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of restaurants by user using the best
    predictor based on a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 9
    ratings = {}
    for restaurant in restaurants:
        name_of_restaurant = restaurant_name(restaurant)
        if restaurant in reviewed:
            ratings[name_of_restaurant]=user_rating(user,name_of_restaurant)
        else:
            ratings[name_of_restaurant]=predictor(restaurant)
    return ratings
    def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    # BEGIN Question 10
    return [restaurant for restaurant in restaurants if query in restaurant_categories(restaurant)]
    # END Question 10


def feature_set():
    """Return a sequence of feature functions."""
    return [lambda r: mean(restaurant_ratings(r)),
            restaurant_price,
            lambda r: len(restaurant_ratings(r)),
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
