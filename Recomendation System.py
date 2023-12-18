
import pandas as pd

data = pd.read_csv('video_data.csv')
credits = pd.read_csv('credits.csv')
movies = pd.read_csv('movies.csv')

#--------------------------FOR NEW USERS----------------------
data = pd.read_csv('video_data.csv', low_memory=False)

#for i in data.columns:
#    print(i)

#print(data.head(4))
#data.info() #Tell about how many entiries in your data file

meanvote = data['vote_average'].mean()
#print(meanvote)

minimumvote = data['vote_count'].quantile(0.90)
#print(minimumvote)

q_video = data.copy().loc[data['vote_count'] >= minimumvote]
q_video.shape


#calculate weight Average
def weighted_rating(x, minimumvote=minimumvote, meanvote=meanvote):
    voters = x['vote_count']
    avg_vote = x['vote_average']
    return(voters/(voters+minimumvote) * avg_vote) + (minimumvote/(minimumvote+voters) * meanvote)

#calculate score using weight average
q_video['score'] = q_video.apply(weighted_rating, axis=1)

#set in decending order
q_video = q_video.sort_values('score', ascending=False)

#pd.set_option('precision', 2)

print(q_video[['title', 'vote_count', 'vote_average', 'score']].head(20))

videomat = q_video.pivot_table(index='user_id',columns='title',values='vote_average')
#pd.set_option('display.max_columns',None)
#print(videomat.head(5))

starwars_user_ratings = videomat['¡Three Amigos!']
print(starwars_user_ratings.head())



#-------------------------- Muskan ----------------------
movies = movies.merge(credits,on='title')

#genres, movie_id, keywords, title, overviews, cast, crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

#Remove movies without overview
movies.isnull().sum()
#drop Movies without overview
movies.dropna(inplace=True)
#Remove duplicate movies
movies.duplicated().sum()

#convert genres, keywords, cast, crew column in simple String list
import ast
def convert(object):
    L = []
    for i in ast.literal_eval(object):
        L.append(i['name']) 
    return L
movies['genres'] = movies['genres'].apply(convert)
#---Write to see whole column:    pd.set_option('display.max_columns',None)
movies.head()

# same do with keyword
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

#same function reapeat for cast to extract actor name
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L
movies['cast'] = movies['cast'].apply(convert)
movies.head()

#same function reapeat for cast to extract director name
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 
movies['crew'] = movies['crew'].apply(fetch_director)

#covert overview from string to list to merge them
movies['overview'] = movies['overview'].apply(lambda x:x.split())

#remove whiteapace from all keywords
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

#make a new column tag and concetinat(merge) cast, crew, genres, keywords all
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#convert backin string
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
#convert it in lower
new['tags'] = new['tags'].apply(lambda x:x.lower())
new.head(5)
#To print onluy tags column:  print(new['tags'][0])


#------------------TEXT VECTORIZATION-----------------

#import sklearn library Class CountVector
from sklearn.feature_extraction.text import CountVectorizer
#create object and use parameter
cv = CountVectorizer(max_features=8000,stop_words='english')

#convert into numpy array
vector = cv.fit_transform(new['tags']).toarray()
vector.shape

#calculte similarity between videos
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
similarity

#Function in which we pass movie name and it will recommend you similar 5 videos
#-----------new[new['title'] == 'The Lego Movie'].index[0]
#print(new[new['title'] == 'The Lego Movie']. index[0])----------
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:10]:
        print(new.iloc[i[0]].title)
        
recommend('Titanic')



























