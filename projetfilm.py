#Projet 2 en appli streamlit
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import unicodedata

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

from palettable.scientific.sequential import Davos_16


# import du DF

link = "https://raw.githubusercontent.com/Nabuchodonosor7/AppP2/main/Notebook_Etape_1_Terminado.csv"
DF = pd.read_csv(link)
st.write(DF)

#sous titre
st.header("Graphiques:")

#Préparation pour le premier graphique

preparation_du_top = DF[(DF['startYear'] == 2021)]
preparation_du_top = preparation_du_top[(preparation_du_top['averageRating'] >= 7)]
TOP_films_2021 = preparation_du_top[(preparation_du_top['numVotes'] >= 8620)]
TOP_films_2021.sort_values(by='numVotes', ascending = False, inplace = True)
TOP_dix_films_2021 = TOP_films_2021.head(10)

st.write(TOP_dix_films_2021)

#Code pour le graphique 

df = TOP_dix_films_2021
fig = px.bar(df, x="primaryTitle", y="averageRating", title = "Note moyenne des films du top 10 de l'année 2021")

st.plotly_chart(fig, use_container_width = True)

             
#fig.update_layout(
    #xaxis_title="Films",
    #yaxis_title="Note moyenne",
    #xaxis={'categoryorder':'total descending'}
    #)

#Code pour le donut

names = ['Drama', 'Comedy', 'Romance', 'Documentary', 'Crime', 'Action', 'Thriller', 
         'Adventure', 'Biography', 'History', 'Autres']      
taille_des_genres=[39758, 16767, 9085, 7810, 7138, 6645, 4988, 4024, 3368, 2768, 18766]


_, _, autotexts = plt.pie(taille_des_genres, labels=names, colors=Davos_16.hex_colors, autopct='%.0f%%',
        pctdistance=0.85, labeldistance=1.2)
for ins in autotexts:
    ins.set_color('white')

plt.title('TOP 10 des genres de film', fontsize=20)
    
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

st.pyplot(my_circle.figure, clear_figure=True)

#Code pour le runtime

DF["genres"] = DF["genres"].str.split(",")
Ex22 = DF.explode('genres')
genreminute22 = Ex22.pivot_table(values = 'runtimeMinutes', index = 'genres', aggfunc = 'mean')

fig, ax = plt.subplots()
fig = px.bar(genreminute22, y="runtimeMinutes",
             title = "Durée moyenne par genre de film"
            )

fig.update_layout(
    xaxis_title="Genre",
    yaxis_title="Minutes"
    )

st.plotly_chart(fig, clear_figure=True)

#sous titre
st.header("Recommandation:")
       

DF["genres"] = DF["genres"].str.split(",")                     
Ex = DF.explode('genres')
GenreF = pd.concat([Ex['genres'].str.get_dummies()],axis=1)     
GenreF = GenreF.groupby(GenreF.index).sum() 
DFM = pd.concat([DF, GenreF],axis=1)                            
DFM1 = DFM.drop(columns=['actor', 'director'])                  
DFM1.dropna(subset=['startYear'], inplace=True)                 
DFM1.dropna(subset=['genres'], inplace=True)                    
DFM1['startYear'] = DFM1['startYear'].astype(int)               
DFM1['titrePropre'] = DFM1['originalTitle'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower().str.replace(' ', '')
                                                                
DFM1.reset_index(drop= True, inplace = True)

X = DFM1.select_dtypes("number")                                                
scaler = StandardScaler()                                                       
scaler.fit(X)                                                                   
X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
modelNN = NearestNeighbors(n_neighbors=6).fit(X_scaled)

def Recherche():                                                                                                                
    pninput = st.text_input('Recherche', 'testest')                                                                                                
    pninput = unicodedata.normalize('NFKD', pninput).encode('ascii', errors='ignore').decode('utf-8').lower().replace(' ', '')  
    selectionInput = DFM1[DFM1['titrePropre'].str.contains(pninput)]
    selectionInput2 = selectionInput[['originalTitle', 'startYear']]                                                            
    if len(selectionInput2) == 1 :                                                                                              
        R1 = selectionInput2.iloc[0]
        return R1                                                                                                               
    elif len(selectionInput2) >= 2 :                                                                                            
        st.write(selectionInput2.set_index(pd.Index(range(1, len(selectionInput2)+1))).to_string(header=False))
        SecInput = int(st.number_input('Insert a number:', min_value=1, step=1))                                 
        R1 = selectionInput2.iloc[SecInput-1]
        return R1                                                                                                               
    else :
        st.write("Navré, nous n'avons trouvé aucune référence correspondante")                                                     
        return None

def Recommendation():                                                                                                           
    R1 = Recherche()                                                                                                            
    if R1 is not None :                                                                                                         
        indiv_concerne = X_scaled.loc[R1.name].to_frame().T                                                                     
        neigh_dist, neigh_cli = modelNN.kneighbors(indiv_concerne, n_neighbors=6)                                               
        cli_ressem = neigh_cli[0][1:]                                                                                           
        Result = DFM1.iloc[cli_ressem]                                                                                          
        index = pd.Index((range(1, len(Result)+1)))
        Result = Result.set_index(index)                                                                                                
        st.write()
        st.write("Vous avez selectionné :", R1['originalTitle'], '(', R1['startYear'], ')')
        st.write("Nous vous recommandons :", Result[['originalTitle', 'startYear']].to_string(header=False), sep='\n')             
    else :
        pass

Recommendation()      