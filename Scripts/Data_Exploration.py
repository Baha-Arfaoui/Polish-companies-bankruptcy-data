from Scripts.librairies import *



class data_analysis():
    
    def __init__(self,dataset):
        self.dataset=dataset
    
    
    def pie_chart(self, column):
        ''' 
        trace un pie chart pour une colonne donnée 
        
        input : 
            le dataset en question 
        output : 
            None
        
        '''
        labels = self.dataset[column].astype('category').cat.categories.tolist()
        counts = self.dataset[column].value_counts().tolist()
        # Create the pie trace
        trace = go.Pie(
            labels=labels,
            values=counts
        )
        # Create the layout
        layout = go.Layout(
            title='Pie Chart'
        )
        # Create the figure
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(width=600, height=600)
        # Show the figure
        fig.show()
        
    

    

   
        
    def plot_correlation(self,data, method):
        ''' 
        Trace une matrice de corrélation pour les données spécifiées.

        input :
            data (pandas.DataFrame) : Les données pour lesquelles tracer la matrice de corrélation.
            method (str) : La méthode de calcul de la corrélation.

        output :
            None
        
        '''
        correlation_matrix = data.corr(method=method, min_periods=1)
        fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                        x=correlation_matrix.columns,
                                        y=correlation_matrix.index,
                                        colorscale='Viridis'))
        fig.update_layout(title='Matrice de corrélation',width=800, height=800)
        fig.show()


    def box_plot(self,column):
        '''  
            
        Trace un graphique en boîte (box plot) pour la colonne spécifiée.

        Input :
            column (str) : Le nom de la colonne pour laquelle tracer le graphique en boîte.

        Output :
            None
        
        '''
        fig = px.box(self.dataset[column], y=str(column))
        fig.update_layout(title='Box plots des variables par classe', width=600, height=400)


        fig.show()
                
        
    

    def plot_violinplots(self,data, column):
        '''  
            
        Trace un violon plot pour la colonne spécifiée.

        Input :
            column (str) : Le nom de la colonne pour laquelle tracer le violon plot.

        Output :
            None
        
        '''
        fig = px.violin(data, y=column, box=True, points='all')
        fig.update_layout(title='Violin plots des variables par classe', width=600, height=400)

        fig.show()
        
      

    def plot_scatterplots(self,data, x_column, y_column, target_column):
        '''  
            
        Trace un scatter plot pour la colonne spécifiée.

        Input :
            x_column (str) : Le nom de la colonne pour laquelle tracer le scatter plot.
            y_column (str) : Le nom de la colonne pour laquelle tracer le scatter plot.
            data ( pd dataframe)
            target_column : la variable cible

        Output :
            None
        
        '''
        fig = px.scatter(data, x=x_column, y=y_column, color=target_column)
        fig.update_layout(title='Nuage de points des variables', width=600, height=400)

        fig.show()
        


    def plot_histograms(self,data, target_column):
        '''  
        
        Trace un scatter plot pour la colonne spécifiée.

        Input :
            x_column (str) : Le nom de la colonne pour laquelle tracer le scatter plot.
            y_column (str) : Le nom de la colonne pour laquelle tracer le scatter plot.
            data ( pd dataframe)
            target_column : la variable cible

        Output :
            None
        
        '''
        fig = px.histogram(data, x=target_column, color=target_column, barmode='overlay')
        fig.update_layout(title='Histogrammes des variables par classe', width=600, height=400)

        fig.show()


        
    
