
from Scripts.librairies import *


class modeling():
    def __init__(self,dataset,target):
        '''  
        Cette Méthode initialse les différents paramètres : dataset , X , y ansi que les algorithmes à utiliser dans notre modélisation
        
        '''
        self.dataset=dataset        
        # Load and preprocess the data
        self.X = self.dataset.drop(target, axis=1)  # Assuming 'target' column contains the class labels
        self.y = self.dataset[target]

        # Split the data into training and test sets
        
        self.model_dict = {
            'LogisticRegresion' : LogisticRegression(random_state=0),
            'Random Forest': RandomForestClassifier(random_state=3),
            'Decision Tree': DecisionTreeClassifier(random_state=3),
            'K Nearest Neighbor': KNeighborsClassifier(),
            'XGBoost':XGBClassifier()
              }

    def train_test_split(self,test_size: float):
        '''  
        Cette méthode perfome le train_test_split 
        input : test_size = 0.2 , 0.3 , 0.4 etc ... 
        output : X_train, X_test, y_train, y_test
        
        '''
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
         
        return X_train, X_test, y_train, y_test
        
    
    def standardize(self,X_train,X_test):
        '''  
        Cette méthode a pour but de transfomer tous les variables numériques ( standscaler ) pour les mettre tous dans le même échelle afin d'améliorer
        les performances des modèles 
        output : X_train_scaled,X_test_scaled 
        '''
        # Scale the numeric features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        return X_train_scaled,X_test_scaled
    

    def train_models(self,X_train,y_train,X_test,y_test,SMOTE=bool):
        ''' 
        Cette méthode va enchainer l'entrainement des différents modèles , comparer les résultats selon la metric de F1_score ( umbalanced data )
        
        input : 
            X_train,y_train,X_test,y_test
        ouput : 
            model_comparison_df : Un dataframe qui résumé les performances de chaque modèle
            best_model : tha champio model 
        '''
        best_model = None
        best_score_f1 = 0
        best_score_acc = 0
        model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []
        if SMOTE:
            for k,v in self.model_dict.items():
                model_name.append(k)
                v.fit(X_train, y_train)
                pickle.dump(v,open('./Artifacts/'+k+'_Smote.pkl','wb'))
                y_pred = v.predict(X_test)
                ac_score_list.append(accuracy_score(y_test, y_pred))
                p_score_list.append(precision_score(y_test, y_pred, average='macro'))
                r_score_list.append(recall_score(y_test, y_pred, average='macro'))
                f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
                model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
                model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
                model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)
        else :      
            for k,v in self.model_dict.items():
                model_name.append(k)
                v.fit(X_train, y_train)
                pickle.dump(v,open('./Artifacts/'+k+'.pkl','wb'))
                y_pred = v.predict(X_test)
                ac_score_list.append(accuracy_score(y_test, y_pred))
                p_score_list.append(precision_score(y_test, y_pred, average='macro'))
                r_score_list.append(recall_score(y_test, y_pred, average='macro'))
                f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
                model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
                model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
                model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)
        model_name=model_comparison_df.iloc[0,0]
        model=self.model_dict[model_name]
        best_model = model
        best_score_f1 = model_comparison_df.iloc[0,4]
        best_score_acc = model_comparison_df.iloc[0,1]
        print(f"Champion model: {best_model.__class__.__name__} , F1_score : {best_score_f1} Accuracy : {best_score_acc}")
        
        return model_comparison_df,best_model
    
    def dump_best_model(self,model, frs, label, out_dir):
        model.fit(frs, label)
        pickle.dump(model, open(out_dir,'wb'))
    
    def model_tuning(self,best_model,X_train,y_train):
        '''  
        Cette Méthode a pour objectif le hyperparameter tuning du champion model 
        
        input : 
            best_model,X_train,y_train
        output : 
            best_model_tuned
        '''

        # Hyperparameter tuning
        param_grid = {}

        if isinstance(best_model,RandomForestClassifier):
            param_grid = {
               
                'criterion': ['gini', 'entropy'],
                'max_depth': [3,5,10]
                
            }
        elif isinstance(best_model, LogisticRegression):
            param_grid = {
                'penalty': ['l1', 'l2'],
                'C': [0.1, 1.0, 10.0],
            
            }
        
        elif isinstance(best_model, KNeighborsClassifier):
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'p': [1, 2]  
            }
        elif isinstance(best_model, XGBClassifier):
            param_grid = {
               
                'max_depth': [3, 5,10],
                'learning_rate': [0.1, 0.01, 0.001]
               
            }
        elif isinstance(best_model, DecisionTreeClassifier):
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [ 3,5, 10]
            }

        if param_grid:
            grid_search = GridSearchCV(best_model, param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            
            best_model_tuned = grid_search.best_estimator_
            print(f"Tuned champion model: {best_model_tuned.__class__.__name__}")
        else:
            print("Hyperparameter tuning not supported for the champion model.")

        return best_model_tuned
    

    def plot_curves(self,in_dir, X_test, y_test, title):
        '''  
        Trace la courbe de précision-rappel pour un modèle donné.

        input :
            in_dir (str) : Le chemin du répertoire où les modèles sont enregistrés.
            X_test (array-like) : Les données de test utilisées pour l'évaluation.
            y_test (array-like) : Les étiquettes réelles correspondant aux données de test.
            title (str) : Le titre du graphique.

        output :
            None
        
        '''
        model = pickle.load(open(in_dir,'rb'))
        y_probas = model.predict_proba(X_test)
        plot_precision_recall(y_test, y_probas,
                            title=str('Precision-recall curve micro-averaged over all classes , Model :  ' + title))
        plt.show()

