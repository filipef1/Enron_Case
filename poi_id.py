#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
from numpy.core.umath_tests import inner1d
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# Apenas para não exibir as mensagens de atenção do F1-score
# por haver de poucos dados de treinamento
import warnings
warnings.filterwarnings('ignore')

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# RESPOSTA: Adicionando recursos para avaliação
features_list = ['poi', 'bonus',  'deferral_payments', 'deferred_income', 'director_fees',
                 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person',
                 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other', 
                 'restricted_stock', 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi',
                 'to_messages', 'total_stock_value', 'total_payments']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


# Armazenando em um Dataframe Pandas para facilitar a análise
df = pd.DataFrame.from_dict(data_dict, orient='index')
# Atualizando os valores com 'NaN' para nulos
df.replace(to_replace='NaN', value=np.nan, inplace=True)

# EXPLORANDO DADOS
# Total de Registros -1 para não contar o cabeçalho
total_r = df.shape[0] -1
print('\nTotal de Registros no Dataset:', total_r)

# Verifica se possui itens duplicados
if df.duplicated().sum() > 0:
    print('{} itens duplicados'.format(df.duplicated().count()))
else:
    print('O dataset não possui itens duplicados')  

# Verificando o total de POIs
total_p = df['poi'].sum()
print('Total de POIS:', total_p)

# Verificando o total de Não POIs
total_np = total_r - total_p
print('Total de Não-POIS:',total_np)

# Verificando o total de características
print('Total de Catacterísticas:', df.shape[1])

# Verificando o total de valores ausentes por característica
ausentes = df.isna().sum()
# Convertendo em um dataframe para facilitar a manipulação
ausentes = pd.Series.to_frame(ausentes)
ausentes.columns = ['total_a']
ausentes['prop_a'] = (ausentes['total_a'] / df.shape[0]) * 100
ausentes = ausentes.sort_values(['prop_a'], ascending=False)
print('\nValores ausentes e proporção de itens faltantes:\n', ausentes)

# Criando um gráfico de barras para melhor vizualização
ausentes['prop_a'].plot(kind='bar', fontsize=10, figsize=(10,4), rot=90, color='darkblue')
plt.title('Proporção de Valores em Faltantes por Característica', fontsize=12)
plt.xlabel('Característica',fontsize=12)
plt.ylabel('Proporção (%)',fontsize=12) 
plt.show()


### Task 2: Remove outliers

# Verifica e imprime os registros com todos valores ausentes
index = 0
print('\nRegistros com todos valores em branco:')
for total in pd.isnull(df).sum(axis = 1):
    if total == len(df.columns)-1:
        print(df.index[index])
    index += 1

# Criando um gráfico de valores médios para procurar outliers
sns.boxplot(data=df, orient="h")
plt.title('Valores Médios e Outliers por Característica', fontsize=12)
plt.xlabel('Pontos Médios',fontsize=12)    
plt.ylabel('Característica',fontsize=12)
plt.show()

# Imprimindo para identificar o outlier mostrado no grafico
print('\nOutliers com "total_stock_value" acima de 4e8:\n', df.index[df['total_stock_value'] >= 400000000])

# RESPOSTA: Removendo Outliers
try:
    data_dict.pop('TOTAL')
    data_dict.pop('LOCKHART EUGENE E')
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    print('\nOutliers excluídos com sucesso!')
except:
    print('\nErro ao excluir os outliers')


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Extraindo rótulos e características do dataset
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# RESPOSTA: Criando novos recursos
for person in my_dataset.values():
    person['prop_bonus'] = 0
    person['prop_from_poi'] = 0
    person['prop_to_poi'] = 0
    
    # Criando uma variável com a proporção de bonus recebido em relação ao salário
    if float(person['bonus']) > 0:
        person['prop_bonus'] = float(person['bonus']) / float(person['salary'])
        
    # Criando uma variável com mensagens recebidas de POIs em relação ao total
    if person['to_messages'] != 'NaN' and person['from_poi_to_this_person'] != 'NaN':
        person['prop_from_poi'] = float(person['from_poi_to_this_person']) / float(person['to_messages'])

    # Criando uma variável com mensagens enviadas a POIs em relação ao total
    if person['from_messages'] != 'NaN' and person['from_this_person_to_poi'] != 'NaN':
        person['prop_to_poi'] = float(person['from_this_person_to_poi']) / float(person['from_messages'])


new_features = features_list + (['prop_bonus', 'prop_from_poi', 'prop_to_poi'])


# Criando uma função com o algoritmo K-Best para selecionar os melhores recursos
def best_features(n_features, l_features, show):
    """ 
    Classifica e imprime o número de recursos desejados com melhor pontuação.    
    
    Argumentos:
        n_features: Número de melhores recursos
        l_features: Lista com os recursos que deseja avaliar
        show: 's' ou 'n', 's' imprime os resultados
                
    Retorna:
        Lista com os recursos de maior pontuação
    """
    data = featureFormat(my_dataset, l_features, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    clf_f = SelectKBest(k = n_features)
    clf_f = clf_f.fit(features, labels)
    results_list = zip(clf_f.get_support(), l_features[1:], clf_f.scores_)
    results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
    count = 1
    bf_list = ['poi']
    
    if show == 's':     
        print('\nMelhores Recursos:')
        for r in results_list:
            if r[0]:
                print(count,'-', r[1], ':', r[2])
                count += 1
                bf_list.append(r[1])
    else:
        for r in results_list:
            if r[0]:
                bf_list.append(r[1])

    return bf_list
    

# Testando a função para avaliar as 10 melhores características
best_features(10, new_features, 's')


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.

# Função criada com a mesma fórmula da documentação
# Foi criada para evitar as mensagens de erro pelo dataset ser pequeno


# RESPOSTA: Testando o melhor número de recursos em cada classificador
for test in range(1,4):
    print('\nTeste de Classificadores {}/3:'.format(test))
    scores = []

    for f in range(1, len(new_features)):
        n_features = best_features(f, new_features, 'n')
        data = featureFormat(my_dataset, n_features, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

        # Testando recursos com Naive Bayes
        clf_nb = GaussianNB()
        clf_nb.fit(features_train, labels_train)
        pred_nb = clf_nb.predict(features_test)
        acc_nb = accuracy_score(labels_test, pred_nb)
        f1s_nb = f1_score(labels_test, pred_nb)
        scores.append({'classificador': 'Naive Bayes', 'recursos': f, 'acuracia': acc_nb, 'f1_score':f1s_nb})

        # Testando recursos com Decision Tree     
        clf_dt = DecisionTreeClassifier()
        clf_dt = clf_dt.fit(features_train, labels_train)
        pred_dt = clf_dt.predict(features_test)
        acc_dt = accuracy_score(labels_test, pred_dt)
        f1s_dt = f1_score(labels_test, pred_dt)
        scores.append({'classificador': 'Decision Tree', 'recursos': f, 'acuracia': acc_dt, 'f1_score':f1s_dt})

        # Testando recursos com Adaboost em Decision Tree
        dct_cl = DecisionTreeClassifier(random_state = 11, max_features = "auto", max_depth = None)
        clf_ad = AdaBoostClassifier(base_estimator=dct_cl)
        clf_ad = clf_ad.fit(features_train, labels_train)
        pred_ad = clf_ad.predict(features_test)
        acc_ad = accuracy_score(labels_test, pred_ad)
        f1s_ad = f1_score(labels_test, pred_ad)
        scores.append({'classificador': 'AdaBoost', 'recursos': f, 'acuracia': acc_ad, 'f1_score':f1s_ad})


    # Convertendo os resultados em um Dataframe para facilitar impressão e análise
    scores = pd.DataFrame(scores)  

    # Criando um gráfico de linhas com os resultados
    for cl in (scores['classificador'].unique()):
        plt.plot(scores[scores['classificador'] == cl]['recursos'], scores[scores['classificador'] == cl]['f1_score'], label=cl)

        # Imprimindo os melhores resultados de cada classificador
        print(scores[(scores.classificador == cl) & (scores['f1_score'] == scores[scores.classificador == cl].f1_score.max())].tail(1))

    plt.title('Score-F1 de Classificadores por Número de Recursos - Teste {}/3'.format(test)) 
    plt.xlabel('Número de recursos K-Best')
    plt.ylabel('Score-F1')
    plt.legend()
    plt.show()

    # Informando o classificador e número de recursos com melhor score
    print('\nMelhor Classificador:\n', scores[scores['f1_score'] == scores['f1_score'].max()].tail(1))
    print('-----------------------------------------------\n')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!


# RESPOSTA: Utilizando GridSearchCV para encontrar a melhor combinação e parâmetros e recursos
score_grid = 0
print("Ajustando parâmetros do classificador...\n")

for f in range(2, len(new_features)):
    n_features = best_features(f, new_features, 'n')
    data = featureFormat(my_dataset, n_features, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Informando parametros a serem testados 
    param_grid = {
                  "base_estimator__criterion" : ["gini", "entropy"],
                  "base_estimator__splitter" :   ["best", "random"],
                  "n_estimators": [1, 50, 100],
                  "learning_rate": [1, 0.7, 0.5]
                  }
    
    # Criando os Classificadores
    dct_cl = DecisionTreeClassifier(random_state = 11, max_features = "auto", max_depth = None)
    adb_cl = AdaBoostClassifier(base_estimator = dct_cl)
    
    # Criando o GridSearch
    grid_search = GridSearchCV(adb_cl, param_grid = param_grid, scoring = 'f1')
    grid_search.fit(features_train, labels_train)
       
    if grid_search.best_score_ > score_grid:
        score_grid = grid_search.best_score_
        recursos = f
        clf = grid_search.best_estimator_ 
    
features_list = best_features(recursos, new_features, 'n')    
print('\nClassificador com parâmetros ajustados:\n', clf)
   

# Informando recursos originais para avaliar impacto dos novos
org_features = ['poi', 'bonus',  'deferral_payments', 'deferred_income', 'director_fees',
                 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person',
                 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other', 
                 'restricted_stock', 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi',
                 'to_messages', 'total_stock_value', 'total_payments']
# Executando testes com recursos originais
for test in range(1,4):
    print('\nTeste do Classificador Ajustado - {}/3 - RECURSOS ORIGINAIS'.format(test))
    test_classifier(clf, my_dataset, org_features)


# Utilizando o script de tester.py avaliar a performance final do algoritmo
for test in range(1,4):
    print('\nTeste do Classificador Ajustado - {}/3'.format(test))
    test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
