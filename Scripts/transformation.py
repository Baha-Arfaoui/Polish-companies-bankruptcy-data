from Scripts.librairies import *


class transformation():
    def __init__(self,dataset,target):
        self.dataset=dataset 
        self.target=target
        
    #Remove outliers 
    def remove_outliers(self,data, threshold=1.5):
        """
        Remove outliers from each column of a DataFrame using the interquartile range (IQR) method.
        
        input:
            data (DataFrame): The input DataFrame.
            threshold (float): The threshold value to determine outliers. Default is 3.
            
        Output:
            DataFrame: The DataFrame with outliers removed.
        """
        
        columns=data.columns.tolist()
        columns.remove(self.target)
        for column in columns:
         
            # Calculate the first and third quartiles (Q1 and Q3)
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            
            # Calculate the IQR (Interquartile Range)
            iqr = q3 - q1
            
            # Define the lower and upper bounds for outliers
            lower_bound = q1 - threshold * q1
            upper_bound = q3 + threshold * q3
            
            # Remove the outliers
            outliers_removed = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            
            # Append the column with outliers removed to the cleaned DataFrame
            data[column] = outliers_removed[column]
        
        return data
    
    
    def impute_missing_values(self,data,strategy):
        '''  
        
        Remplace les valeurs manquantes dans les données en utilisant la stratégie spécifiée.

        input :
            data (DataFrame) : Les données contenant les valeurs manquantes à remplacer.
            strategy (str) : La stratégie de remplacement des valeurs manquantes ( mean ).

        Output :
            data_impute_df (DataFrame) : Les données avec les valeurs manquantes remplacées.
        
        '''
        imputation = SimpleImputer(missing_values=np.nan, strategy=strategy)
        data_impute = imputation.fit_transform(data)
        data_impute_df = pd.DataFrame(data_impute, index=data.index, columns=data.columns)
        
        return data_impute_df
    
    
    def anova_test(self,data):
        '''  
        L'ANOVA est un test qui évalue la différence des moyennes entre plusieurs groupes. Dans le cas de variables numériques et d'une variable 
        cible avec plusieurs classes, vous pouvez effectuer une ANOVA à un facteur ou une ANOVA à plusieurs facteurs pour déterminer 
        s'il existe des différences significatives entre les groupes.
        
        input:
            data ( pandas dataframe)
        output:
                results_df : pd.dataframe avec le résultat de l'ANOVA.
        
        '''
        # Separate the numeric variables and the target variable
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        target_variable = self.target
        
                # Perform ANOVA test for each numeric variable
        results = []
        for column in numeric_columns:
            groups = []
            for group in data[target_variable].unique():
                groups.append(data[data[target_variable] == group][column])
            f_value, p_value = f_oneway(*groups)
            results.append({'Variable': column, 'F-value': f_value, 'p-value': p_value})

        # Convert results to a DataFrame for better presentation
        results_df = pd.DataFrame(results)    
        return results_df
    
    
    def select_k_best(self,score, X, Y):
        '''  
        Sélectionne les meilleures variables en fonction du score donné.

        Input :
            score : L'objet score utilisé pour évaluer les variables.
            X (DataFrame) : Les variables d'entrée.
            Y (Series ou array) : La variable cible.

        Output :
            feature_names (list) : La liste des noms des meilleures variables sélectionnées.
        
        '''
        selector = SelectKBest(score, k=20)
        selector.fit_transform(X, Y)
        names = X.columns.values[selector.get_support()]
        scores = selector.scores_[selector.get_support()]
        names_scores = list(zip(names, scores))
        df_reduced = pd.DataFrame(data=names_scores, columns=['Predictor', 'mutual_info'])

        df_reduced = df_reduced.sort_values(['mutual_info', 'Predictor'], ascending=[False, True])
        print(df_reduced)
        return df_reduced.Predictor
    
   