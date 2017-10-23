#-*- coding=utf-8 -*-

###################################################################
# AUTOR:  DAVID CUESTA
# FECHA:  12/10/2017
# NOTAS:
###################################################################

###################################################################
#   AREA DE IMPORTACION DE LIBRERIAS
# 44 importaciones
###################################################################

#importacion de las librerias utilizadas
from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import render_to_response
from django.conf import settings
from django.template import RequestContext, Context, Template
from django.http import HttpResponseRedirect
import random 
import datetime
import nltk  #para el procesamiento del lenguaje natural
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords, cess_esp , conll2002, machado, conll2007
from nltk.probability import FreqDist
import string
from itertools import chain
from nltk.metrics import ConfusionMatrix
from nltk.collocations import *
import paramiko  #para acceso ssh al servidor
import os
import yaml   #para configuracion
from nltk import word_tokenize
from nltk.data import load
from nltk.stem import SnowballStemmer
from string import punctuation
from nltk.tokenize.toktok import ToktokTokenizer
import collections
import nltk.metrics
from nltk.corpus import names
import threading
from nltk.util import in_idle
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus.util import LazyCorpusLoader
import argparse, math, itertools, os.path
import nltk.corpus, nltk.data
from django.utils.datastructures import MultiValueDictKeyError
from django.views.decorators.csrf import csrf_exempt
from googletrans import Translator
from textblob import TextBlob
import pickle  #para grabar la instancia del clasificador como archivo binario
import requests
from django.core.cache import cache #importamos el objeto cache

###################################################################
#           AREA DE DEFINICION DE VARIABLES GLOBALES
# 18 variables globales
###################################################################

'''
Referencia de las variables del archivo YAML
url='http://62.204.199.211:8880'
urlSin='62.204.199.211:8880'
urlSinPuerto='62.204.199.211'
nickname='david'
clave='daviduned2017.'
'''
url=""
urlSin=""
urlSinPuerto=""
nickname=""
clave=""
puerto=22
segundos=20
texto=""
contador=1
non_words=""
stemmer=""
neutral="0"
polar="0"
positiva="0"
negativa="0"
ctexto=""
resulFinal=""
compound=""

global nbclassifier
global known_words
# Declaramos las las rutas de los archivos de lexico y clasificador
classifier_file="../res/train/classifier.pickle"
lexicon_file="../res/lexicon/sdal_lexicon.csv"


##########################################################################
#  AREA DE IMPLEMENTACION DE FUNCIONES  
#  Listado de 19 funciones
##########################################################################
#  1- limpia()                 limpia pantalla adaptada a linux y windows
#  2- index(request,cadenatexto=None) Landing page
#  3- conectarServidor()
#  4- cargaCorpus()
#  5- word_feats(words)
#  6- entrenamiento()
#  7- clasificar()
#  8- resultado()
#  9- punct_features(tokens,i)
#  10-leerYAML()
#  11-test_esp()
#  12-corpus(request)
#  13-gender_features(name)
#  14-stem(word)
#  15-stem_tokens(tokens)
#  16-tokenize(text)
#  17-AnalizadorSentimiento()
#  18-BayesClasificador()
#  19-vadem()
##########################################################################


##########################################################################
# Función 1 limpia pantalla de la consola en windows o linux
##########################################################################
def limpia():
    if os.name == 'posix':
        os.system('clear')

    elif os.name in ('ce', 'nt', 'dos'):
        os.system('cls')

##########################################################################
# Función 2 función para llamar a la pagina landing page del sitio WEB
##########################################################################

@csrf_exempt
def index(request):
	global contador
	global texto
	response = ''
	cadtexto=""
	cadresultado=""
	valor=""
	if request.method == 'GET':
		cadtexto = str(request.POST.get('txtTexto')) #en principio el mejor
		print("\n cadtexto introducido por el usuario= "+str(cadtexto)+"\n")
		cadtexto = str(request.GET.get('txtTexto')) #en principio el mejor
		print("\n cadtexto introducido por el usuario= "+str(cadtexto)+"\n")
	if request.method == 'POST':	
		cadtexto = str(request.POST.get('txtTexto')) #en principio el mejor
		print("\n POST_GET cadtexto introducido por el usuario= "+str(cadtexto)+"\n")
	
	if contador==1:
		limpia()
		print("\n Leyendo el fichero de configuracion \n")
		leerYAML()
		conectarServidor()
		#cargaCorpus()    no activarlo ya estan cargados los corpus
		#entrenamiento()  anulado para pruebas
		#clasificar()     anulado para pruebas
		#resultado()
		contador=contador+1
		bloquetxt="No me gustan las golosinas" #frase de prueba de entrenamiento
		AnalizadorSentimiento(bloquetxt)
		BayesClasificador()
		vader()
		return render_to_response('index.html')
	else:
		limpia()	
		try:
			if request.method == "GET":	
				cadtexto = str(request.GET.get('txt1',""))
				print("\n      GET cadtexto= "+str(cadtexto)+"\n")
			if request.method == "POST":	
				cadtexto = str(request.POST.get('txt1',""))
				print("\n      -POST- \n\n       "+str(cadtexto)+"\n")
		except MultiValueDictKeyError:
			print("\n Error en cadtexto \n")
		
		bloquetxt=cadtexto
		valor=cadtexto
		AnalizadorSentimiento(bloquetxt)
		#clasificarConBayes()		
		global resulFinal
		global ctexto
		global neutral
		global polar
		global positiva
		global negativa
		print("\n valor="+str(valor)+ "\n\n")	
		ctexto=""
		ctexto=ctexto+'======================================= </center><br><br>'
		ctexto=ctexto+'<br><div style="background-color:#A9F5BC"><center><br><h2><b>Subjetividad</b></h2><br><br><h3>'
		ctexto=ctexto+'  Neutral: '+neutral+'<br><br>'
		ctexto=ctexto+'  Polar: '+polar+'<br><br>'
		ctexto=ctexto+'  Positiva:'+positiva+'<br><br>'
		ctexto=ctexto+'  Negativa:'+negativa+'<br><br><br><b></h3><h2>'
		ctexto=ctexto+'  Resultado: '+resulFinal+'<b></h2><br></center></div>'
		ctexto=ctexto+'  <center><br><h4> Frase Analizada: <br><br>'+str(valor)+'</h4></center>'
		print("\n ctexto="+str(ctexto)+ "\n\n")	
		html = "<html><body><br><center><h3>Resultado del Análisis de SENTIMIENTOS:</h3> %s </body></html>" % (ctexto)
		return HttpResponse(html) 

##########################################################################
# Función 3 función para conectar con el servidor ssh
##########################################################################
def conectarServidor():
	print("\n   ==============================================================\n")
	print("         Conectando con el servidor de la Universidad            ")
	print("\n   ==============================================================\n")
	try:
		nltk.set_proxy(url, (nickname, clave))	
	except Exception:
		print("  Error de Comunicacion con el Servidor") 
		return
	print("         Servidor Conectado OK..... \n\n")
	print("       Fase 1/6 Iniciamos la conexion SSH.............. \n\n")
	# Datos para la conexión SSH
	ssh_servidor = urlSin
	ssh_usuario  = nickname
	ssh_clave    = clave
	ssh_puerto   = puerto 
	comando      = 'ls' # comando que vamos a ejecutar en el servidor para comprobar acceso
	# Conectamos al servidor
	try:
		ssh = paramiko.SSHClient()  # Iniciamos un cliente SSH
		ssh.load_system_host_keys()  # Agregamos el listado de host conocidos
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Si no encuentra el host, lo agrega automáticamente
		# Iniciamos la conexión.
		conexion= ssh.connect(urlSinPuerto, port=puerto, username=nickname, password=clave,  timeout=segundos)  
		print("    Conexion cliente establecida con el servidor de la Universidad \n  ")
		print("    "+str(ssh))
		print("\n   ==============================================================")
		print("\n                 Archivos del Servidor ")
		#listar los ficheros y directorios del servidor 
		stdin, stdout, stderr = ssh.exec_command('ls -l')
		x = stdout.readlines()
		print("\n   ==============================================================\n")
		for line in x:
			print("      "+str(line))
	except paramiko.ssh_exception.SSHException:
		print("Error de Comunicacion con el Servidor SSH") 
		ssh.close()
		return
	except paramiko.AuthenticationException:
		print("Error de Autenticación con el Servidor SSH") 
		ssh.close()
		return
	except socket.error, e:
		print("Error de Socket con el Servidor SSH") 
		ssh.close()
		return
	print("\n                    Servidor SSH en Escucha........... ")
	print("\n   ==============================================================\n")	
	#conexion.close()	
	return
	
##########################################################################
# Función 4 función para cargar en memoria el corpus
# el corpus es un conjunto linguisitico amplio y estructurado de ejemplos 
# reales del uso de la lengua
##########################################################################	
def cargaCorpus():
	print("         Fase 2/6  Preparar archivos de Corpus \n")
	print("\n         Procesando el dataset en memoria ===> cess_esp")
	print("\n         espere un momento por favor...................\n")
	#no imprimirlo es muy grande
	reviews = list(cess_esp.words()) 
	new_train, new_test = reviews[0:100], reviews[101:200]	
	print(new_train)
	print("\n\n\n           Test sobre el siguiente conjunto de prueba de cess_esp...... \n")
	print(new_test)
	return

##########################################################################
# Función 5 función para extraccion de caracteristicas de diccionarios 
# feaststructs  
##########################################################################	
def word_feats(words):
	return dict([(word, True) for word in words])

##########################################################################
# Función 6 función para entrenar y probar el clasificador de naive bayes 
# con el corpus asociado
##########################################################################	
def entrenamiento():
	global texto
	print("\n   ==============================================================\n")
	print("\n          Fase 3/6 Clasificacion y Entrenamiento \n")
	#creacion de la lista de tuplas formada por pares de valores
	conjunto = [('Me gusta el cine', 'pos'),
    ('me gusta comer en restaurantes de Madrid', 'pos'),
    ('si me gustan la peliculas de accion', 'pos'),
    ('No me gusta el teatro', 'neg'),
    ('No me gusta la poesia moderna', 'neg'),
	]

	test = [('Quiero ir a un restaurante a comer', 'pos'),
    ('Normalmente me gusta el cine', 'neg'),
    ('la pelicula del cine bristol no me gusta', 'neg'), 
	('me gusta el cine', 'pos'),
    ('Me gustan los libros de historia', 'pos'), 
	]
	
	#obtenemos el clasificador
	print("\n")
	print("        Clasificando con el algoritmo Naive Bayes ....... \n")
	exactitud=""
	cadena=""
	
	negids = movie_reviews.fileids('neg')
	posids = movie_reviews.fileids('pos')
	negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
	posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
	negcutoff = len(negfeats)*3/4
	poscutoff = len(posfeats)*3/4
	trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
	testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
	print('        Entrenamiento sobre %d instancias, Test de prueba sobre %d instancias' % (len(trainfeats), len(testfeats)))
	classifier = NaiveBayesClassifier.train(trainfeats)
	print("\n")
	exactitud=str(nltk.classify.util.accuracy(classifier, testfeats))
	cadena='        Precision: '+exactitud
	print(cadena)
	print("\n   ==============================================================\n")
	print("\n       Fase 4/6  Tokenizando corpus \n")
	print("\n       espere un momento por favor.......\n\n")
	#print("       Resultado del entrenamiento: \n\n  ")
	classifier.show_most_informative_features() #classifier.show_informative_features(5) 
	#classifier.accuracy(test) #puntuacion
	print("\n   ==============================================================\n")
	return

##########################################################################
# Función 7 función para clasificar tokens y staming con el algoritmo 
# naive bayes 
##########################################################################		
def clasificar():	
	global texto
	print("        Fase 5/6 Clasificacion de los tokens y steaming \n")
	print("        espere un momento por favor........ \n")
	stop = stopwords.words('spanish') 
	print("\n        Diccionario castellano de palabras finales: \n\n  "+str(stop))
	all_words = FreqDist(w.lower() for w in movie_reviews.words() if w.lower() not in stop and w.lower() not in string.punctuation)
	print("\n       Palabras clasificadas:   "+str(all_words))
	documents = [([w for w in movie_reviews.words(i) if w.lower() not in stop and w.lower() not in string.punctuation], i.split('/')[0]) for i in movie_reviews.fileids()]
	word_features = FreqDist(chain(*[i for i,j in documents]))
	word_features = word_features.keys()[:100]
	numtrain = int(len(documents) * 90 / 100)
	print("\n       Num. Entrenamientos de la clasificacion ==>     "+str(numtrain))
	train_set = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag in documents[:numtrain]]
	test_set = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag in documents[numtrain:]]
	classifier = NaiveBayesClassifier.train(train_set)
	print("\n       Precision ==>     "+str(nltk.classify.accuracy(classifier, test_set))+" \n")
	classifier.show_most_informative_features(5)
	texto=str(classifier)
	# inicializar el extractor de raices lexicales
	print("\n         Inicializando el extractor de raices lexicales \n")
	global stemmer
	stemmer = SnowballStemmer('spanish')
	print("         Stemmer - extracion de raices semanticas ==> "+str(stemmer))
	
	# inicializar la lista de palabras ignoradas 
	global non_words
	non_words = list(punctuation)  
	print("\n          Lista de Palabras ignoradas:    \n")
	print("\n          "+str(non_words))
	print(" \n         Agregando signos de apertura y digitos \n")
	non_words.extend(['¿', '¡'])  
	non_words.extend(map(str,range(10)))
	#frase de prueba
	t="Me gusta ver el amanecer pero no me gusta el anochecer"
	print(tokenize(str(t)));
	print("\n\n        Tokenizacion finalizada ")
	print("\n   ==============================================================\n")
	return
	
##########################################################################
# Función 8 el resultado final
##########################################################################		
def resultado():
	new_text = ['En', 'un', 'lugar', 'de', 'la', 'Mancha', ',', 'de', 'cuyo', 
	'nombre', 'no', 'quiero', 'acordarme', ',', 'no', 'ha', 'mucho', 'tiempo', 
	'que', 'vivía', 'un', 'hidalgo', 'de', 'los', 'de', 'lanza', 'en', 'astillero',
	',', 'adarga', 'antigua', ',', 'rocín', 'flaco', 'y', 'galgo', 'corredor', '.',
	'Una', 'olla', 'de', 'algo', 'más', 'vaca', 'que', 'carnero', ',', 'salpicón', 
	'las', 'más', 'noches', ',', 'duelos', 'y', 'quebrantos', 'los', 'sábados', 
	',', 'lantejas', 'los', 'viernes', ',', 'algún', 'palomino', 'de', 'añadidura',
	'los', 'domingos', ',', 'consumían', 'las', 'tres', 'partes', 'de', 'su', 
	'hacienda', '.', 'El', 'resto', 'della', 'concluían', 'sayo', 'de', 'velarte', 
	',', 'calzas', 'de', 'velludo', 'para', 'las', 'fiestas', ',', 'con', 'sus', 
	'pantuflos', 'de', 'lo', 'mesmo', ',', 'y', 'los', 'días', 'de', 'entresemana',
	'se', 'honraba', 'con', 'su', 'vellorí', 'de', 'lo', 'más', 'fino', '.']
	
	print("\n           Fase 6/6  Resultado final")
	sentences = cess_esp.sents()
	tokens = []
	boundaries = set()
	offset = 0
	for sent in sentences:
		tokens.extend(sent)
		offset += len(sent)
		boundaries.add(offset-1)
		
	featureset = [(punct_features(tokens, i), (i in boundaries))
		for i in range(1, len(tokens)-1)
		if tokens[i] in '.?!']
	size = int(len(featureset) * 0.1)
	train_set, test_set = featureset[size:], featureset[:size]
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	nltk.classify.accuracy(classifier, test_set)	
	#segment_sentences(new_text)
	
	return

##########################################################################
# Función 9  función de apoyo en la tokenización
##########################################################################	
def punct_features(tokens,i):
	return {'next-word-capitalized': tokens[i+1][0].isupper(),
            'prev-word': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i-1]) == 1}

##########################################################################
# Función 10  función de carga de los parámetros de configuración
##########################################################################	
def leerYAML():
	global url
	global urlSin
	global urlSinPuerto
	global nickname
	global clave
	global puerto
	
	url='http://62.204.199.211:8880'
	urlSin='62.204.199.211:8880'
	urlSinPuerto='62.204.199.211'
	nickname='david'
	clave='daviduned2017.'
	puerto=22
	print("\n       Directorio actual ==> "+str(os.getcwd()))
	print("\n       Cargando el archivo YAML de configuracion \n")
	archivo="App.yaml"
	stream = open(archivo, "r")
	docs = yaml.load_all(stream)
	for doc in docs:
		for k,v in doc.items():
			print(k, "->", v)
			if k==url:
				url=v
			if k==urlSin:
				urlSin=v
			if k==urlSinPuerto:
				urlSinPuerto=v
			if k==nickname:
				nickname=v	
			if k==clave:
				clave=v	
			if k==puerto:
				puerto=v	
		print("\n ,") 

##########################################################################
# Función 11  función de test del corpus español de prueba
##########################################################################	
def test_esp():
        words = cess_esp.words()[:15]
        txt1 = "El grupo estatal Electricité_de_France -Fpa- EDF -Fpt- anunció hoy , jueves , la compra del"
        self.assertEqual(words, txt1.split())

##########################################################################
# Función 12  función de descarga de corpus
##########################################################################	
def corpus(request):
	print("descargando corpus \n")
	nltk.download()
	#nltk.download("movie_reviews")
	nltk.download("spanish_grammars")
	nltk.download("cess_esp")
	print("FIN \n")

##########################################################################
# Función 13  función de conversión a minúsculas de los textos
##########################################################################	
def gender_features(name):
	features = {}
	features["first_letter"] = name[0].lower()
	features["last_letter"] = name[-1].lower()
	features["first_three_letter"] = name[:3].lower()
	features["last_three_letter"] = name[-3:].lower()
	return features
 
##########################################################################
# Función 14  funcion de extraccion de raices lexicales de una palabra
##########################################################################	
def stem(word):
	global stemmer
	return stemmer.stem(word)

##########################################################################
# Función 15  funcion de extraccion de raices lexicales sobre una lista
##########################################################################	
def stem_tokens(tokens):  
	stemmed = []
	for item in tokens:
		stemmed.append(stem(item))
	return stemmed

##########################################################################
# Función 16  funcion de disgregacion de palabras - separa palabras
##########################################################################	
def tokenize(text): 
	global non_words 
	print("\n         Palabras eliminadas: \n")
	print("\n         "+str(non_words)+" \n")
	qgrams=[];
	trigrams=[];
	bigrams=[];
	text=text.lower()
	print("\n         Texto en minusculas de prueba a tokenizar: \n")
	print("\n         "+str(text)+" \n")
	text = ''.join([c for c in text if c not in non_words])
	tokens =  word_tokenize(text)
	tokens = stem_tokens(tokens)
	return tokens	
	
##########################################################################
# Función 17  funcion del analizador de sentimiento 
##########################################################################	
def AnalizadorSentimiento(bloquetxt):
	print("\n       Analizador de Sentimiento: \n ")
	global neutral
	global compound
	global positiva
	global negativa
	global polar
	global resulFinal
	auxiliar=""
	bloquelocal=bloquetxt
	print("\n\n     le llega como bloque.............: "+str(bloquelocal)+"\n")
	
	frase = ["Great place to be when you are in Bangalore.",
	"The place was being renovated when I visited so the seating was limited.",
	"Loved the ambience, loved the food",
	"The food is delicious but not over the top.",
	"Service - Little slow, probably because too many people.",
	"The place is not easy to locate",
	"Mushroom fried rice was tasty"]
	
	conjunto = [('Me gusta el cine'),
    ('me gusta comer en restaurantes de Madrid'),
    ('si me gustan la peliculas de accion'),
    ('No me gusta el teatro'),
    ('No me gusta la poesia moderna'),
	('Si me gusta la poesia moderna'),
	]
	stop = stopwords.words('spanish') 
	print("\n        Diccionario castellano de palabras finales: \n\n  "+str(stop)+"\n\n")
	eee=nltk.corpus.cess_esp.fileids()
	print("\n          fileids del corpus cess_esp: \n\n      "+str(eee)+"\n\n")
	print("\n        cess_esp: \n\n  "+str(nltk.corpus.cess_esp.words())+"\n\n")
	#conjunto1 = [("No me gusta el cine y no me gusta comer en restaurantes de la zona de Villalba ni me gustan la peliculas de accion y ni me gusta el teatro")]
	#conjunto1 = [("si me gusta siempre el cine y si me gusta comer siempre en el restaurante japones sure de la zona de Villalba y si me gustan la peliculas de accion y si me gusta el teatro")]
	conjunto1=bloquelocal.split()
	print("\n   ====================== bloque de frases en castellano a procesar ========================================\n\n")
	pos=0.0
	neg=0.0
	neu=0.0
	com=0.0
	
	sid = SentimentIntensityAnalyzer()
	for sentence in conjunto1:
		print(sentence)
		ss = sid.polarity_scores(sentence)
		for k in ss:
			print("{0}: {1} , ".format(k, ss[k]))
			if k=="neg":
				auxiliar=ss[k]
				neg=float(auxiliar)
				negativa=str(neg)
			if k=="pos":
				auxiliar=ss[k]
				pos=float(auxiliar)
				positiva=str(pos)
			if k=="neu":
				auxiliar=ss[k]
				neu=float(auxiliar)
				neutral=str(neu)
			if k=="compound":
				auxiliar=ss[k]
				com=float(auxiliar)
				compound=str(com)
				polar=compound
		print()
	if neg>pos:
		resulFinal="Negativo"		
	elif pos>neg:
		resulFinal="Positiva"		
	else:
		resulFinal="Neutro"		
	print("neg="+str(negativa)+" posi="+str(positiva)+" neu="+str(neutral)+" compoun="+str(compound)+"\n")	
	print("\n   ==============================================================\n")

##########################################################################
# Función 18  funcion del clasificador de bayes 
##########################################################################		
def BayesClasificador():
	print("\n           inicio clasificador")
	# Step 1 – Training data
	train = [("Great place to be when you are in Bangalore.", "pos"),
	("The place was being renovated when I visited so the seating was limited.", "neg"),
	("Loved the ambience, loved the food", "pos"),
	("The food is delicious but not over the top.", "neg"),
	("Service - Little slow, probably because too many people.", "neg"),
	("The place is not easy to locate", "neg"),
	("Mushroom fried rice was spicy", "pos"),
	]
	# Step 2
	dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
	# Step 3
	t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
	# Step 4 – the classifier is trained with sample data
	classifier = nltk.NaiveBayesClassifier.train(t)
	test_data = "Manchurian was hot and spicy"
	test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}
	print("\n\n    Resultado final:   "+str(classifier.classify(test_data_features)))
	global resulFinal
	resulFinal=str(classifier.classify(test_data_features))
	print("\n   ==============================================================\n")
	return
	
##########################################################################
# Función 19  funcion del analizador de sentimiento vader
##########################################################################		
def vader():
	text=["i have a good feeling about this."]
	text1='she has the worse character in the class'
	sid=SentimentIntensityAnalyzer()
	for word in text:
		ss = sid.polarity_scores(word)
		print(str(ss))
	nltk.sentiment.util.demo_vader_instance(text1)
	nltk.sentiment.util.demo_liu_hu_lexicon(text1, plot=False)
	print("\n   ==============================================================\n")
	text2 = ["me gusta comer en restaurantes de Madrid"]
	text3 = "El Jefe tiene una buena amistad con sus empleados"
	sid=SentimentIntensityAnalyzer()
	for word in text2:
		ss = sid.polarity_scores(word)
		print(str(ss))
	nltk.sentiment.util.demo_vader_instance(text3)
	nltk.sentiment.util.demo_liu_hu_lexicon(text3, plot=False)
	print("\n   =====  pulsa una tecla para iniciar el sitio web======\n")
	#t=raw_input();
	return
	
'''	
@csrf_exempt
def indice(request, txtTexto):
	print("\n =====> entra en indice \n")
	metodo=request.method
	print("\n =====> metodo usado "+str(metodo)+"\n")
	print("\n =====> txtTexto "+str(request.POST['txtTexto'])+"\n")
	print("\n =====> txtTexto "+str(request.GET.get('txtTexto', None))+"\n")
	print("\n =====> txtTexto "+str(request.POST.get('txtTexto', None))+"\n")
	
	if request.method=="POST":			
		print("OKKKK \n")
		print("request.POST="+str(request.POST)+"\n")
		print("request.POST="+str(request.POST('txtTexto'))+"\n")	
		print("request.POST="+str(request.GET('txtTexto'))+"\n")	
		print("request.POST="+str(request.GET.get('txtTexto',''))+"\n")	
	#return render_to_response('resultado.html',{'listado': listado})		
'''

###########################################################################################
	
# funcion para extraer caracteristicas para el clasificador
def extract_feature(w):
	return {"word":w}

def clasificarConBayes():	
	# funcion que construye el set para entrenamiento del algoritmo clasificador
	# lexico para el entrenamiento
	print("Clasificar con bayes final=\n")	
	lexicon_lines=open(lexicon_file,"r", encoding='utf-8').readlines();
	labeled_words=[]
	known_words=[]
	for line in lexicon_lines:
		split=line.replace("\n","").split(";")
		word=split[0].split("_")
		word=stem(word[0])
		known_words.append(word)
		labeled_words.append((word,split[1]))
	feature_set=[(extract_feature(n), sentiment) for (n, sentiment) in labeled_words]

	# Para evitar crear y entrenar el clasificador una y otra vez 
	# usamos pickle para grabar el objeto entrenado como archivo binario
	if not (os.path.isfile(classifier_file)):
		nbclassifier = nltk.NaiveBayesClassifier.train(feature_set)
		f = open(classifier_file, 'wb')
		pickle.dump(nbclassifier, f)
		f.close();
	else:
		f = open(classifier_file, 'rb')
		nbclassifier = pickle.load(f)
		f.close()
	print("Ingrese una frase para el análisis:")
	t=raw_input()
	exp_count=0;
	exp_value=0;
	tokens=tokenize(t)
	for e in tokens:
		if e in known_words:
			exp_count+=1
			sentiment=nbclassifier.classify(extract_feature(e))
			exp_value+=float(sentiment)-1
	if exp_count>0:
		scale_sentiment=exp_value/exp_count
		acceptance=scale_sentiment/2*100 
		print("Valor de Sentimiento: "+str(scale_sentiment+1))
		print("Aceptacion: "+str(acceptance)+"%")
	else:
		print("No se pudo determinar")	
	return	