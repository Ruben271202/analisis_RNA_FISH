import importlib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import radius_neighbors_graph
import alphashape
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import umap
from scipy import stats
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def cluster_and_plot_scaled(df_completo, genes_clustering, genes_color_dic,nombre_foto, show_contour = True, 
                            tipos_plot = [],  peso_espacial=15, distance_threshold=2.5, alpha_val=0.1):
    ''' 

    Esta función acepta un data_frame de varios tránscritos y los agrupa teniendo en cuenta la información de la 
    posición y de la expresión génica mediante el método de agglomerative clustering. Posteriormente, cada uno de los clústers 
    se representa y se dibuja un contorno a su alrededor del color del gen marcador dominante. 

    Parámetros:

    - df_completo:'data frame' crudo que contiene dos columnas con las coordenadas X e Y de los tránscritos de RNA FISH 
    y las 100 columnas de genes. 
    Cada fila es una coordenada de la imagen en la que se expresa o no cada gen, indicado con 1 y 0 respectivamente. 
    - genes_clustering: lista con genes que se quieran usar para el clustering. Se seleccionan las filas en las que 
    se expresan del 'data frame' anterior y se realiza el clustering con esa información. 
    - genes_color_dict: diccionario cuyas claves son genes cuyos clusters queremos representar, siendo los valores 
    el color asignado a cada gen. 
    - peso espacial: número por el que se multiplica a las coordenadas normalizadas para agregarles peso en el clustering.
    - tipos_plot: una lista con los tipos celulares cuyos clústers se quieren representar 
    - distance_treshold: distancia máxima en la que dos clústers generados en 'agglomerative clustering' se fusionan. Un valor mayor 
    resulta 
    - alpha_val: Indica que tan ajustado o flexible es el contorno que se dibuja alrededor de los puntos del clúster


    Resultados:

    - Plot de puntos que representan los tránscritos, coloreados según el tipo celular al que se pertenecen. Se dibuja también un
    contorno con alphashape que incluye todos los tránscritos pertenecientes al mismo clúster. 
    - df_clustering: Data frame similar al data frame input, con la inclusión de una columna que indica el número de clúster al que 
    pertenece el tránscrito. 
    '''

    #lista de genes para la clusterizacion que se encuentran en el data frame de expresión
    genes_existentes = [g for g in genes_clustering if g in df_completo.columns] 
    if not genes_existentes:
        raise ValueError("Ninguno de los genes indicados está presente en el archivo.")

    #se eliminan filas del original donde no esta expresado un gen de interes
    df_clustering = df_completo[df_completo[genes_existentes].astype(int).sum(axis=1) > 0].copy() 

    coords = df_clustering[['x', 'y']].values #solo las dos columnas de las coordenadas
    gene_cols = df_clustering.drop(columns=['x', 'y']).select_dtypes(include=[np.number]).columns #lista de los nombres de las columnas de los genes
    genes = df_clustering[gene_cols].values  #solo las columnas de la expresión de genes, sin las coordenadas

    # Escalar por separado
    # La normalización del standardscaler es X-media/std y por columnas
    scaler_coords = StandardScaler()
    coords_scaled = scaler_coords.fit_transform(coords)

    scaler_genes = StandardScaler()
    genes_scaled = scaler_genes.fit_transform(genes)

    # Ponderar espacial. Se permite modificar la importacia de que estén próximos los puntos
    coords_scaled *= peso_espacial

    # Combinar
    features = np.hstack([coords_scaled, genes_scaled])

    # Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average'
    )
    labels = clustering.fit_predict(features)
    df_clustering['cluster'] = labels

    print("Total number of clusters:", len(set(labels)))

    # Plot
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('white')

    # Genes a representar
    if tipos_plot:
        dict_plot_color = {
            gen: color
            for tipo, subdic in genes_color_dic.items()
            if tipo in tipos_plot
            for gen, color in subdic.items()
        }
    else:
        dict_plot_color = {
            gen: color
            for tipo, subdic in genes_color_dic.items()
            for gen, color in subdic.items()
        }
    for gene, color in dict_plot_color.items():
        if gene not in df_completo.columns:
            continue
        sg = df_clustering[df_clustering[gene] == 1]
        ax.scatter(sg['x'], sg['y'], s=0.5, color=color, label=gene, alpha=0.5) #únicamente se plotean los genes marcadores en color

    genes_plot = [g for g in dict_plot_color.keys() if g in df_completo.columns]
    if not genes_plot:
        raise ValueError("Ninguno de los genes indicados está presente en el archivo.")
    
    df_plot = df_clustering[df_clustering[genes_plot].astype(int).sum(axis=1) > 0].copy() #cuando el clustering es con 100, nos quedamos con las filas de los 36 interesantes
    gene_cols = df_plot.drop(columns=['x', 'y', 'cluster']).select_dtypes(include=[np.number]).columns
    clusters_plot = df_plot['cluster']
    print(f'Nº de clústers: {len(set(clusters_plot))}')#clusters que contienen genes del diccionario de colores

    if show_contour == True:
        # Contornos por cluster
        for cl in sorted(df_plot['cluster'].unique()):
            cluster_points = df_plot[df_plot['cluster'] == cl]
            if len(cluster_points) < 8:
                continue
            
            # Gen dominante en el cluster
            gene_counts = {gene: cluster_points[gene].sum() for gene in gene_cols}
            gen_dominante = max(gene_counts, key=gene_counts.get)
            color_contorno = dict_plot_color.get(gen_dominante, 'black')

            # Contorno con alphashape
            try:
                puntos = cluster_points[['x', 'y']].values
                alpha_shape = alphashape.alphashape(puntos, alpha=alpha_val)
                if alpha_shape.is_empty:
                    continue
                if isinstance(alpha_shape, Polygon):
                    xs, ys = alpha_shape.exterior.xy
                    ax.plot(xs, ys, color=color_contorno, linewidth=1.2)
                else:
                    for geom in alpha_shape.geoms:
                        xs, ys = geom.exterior.xy
                        ax.plot(xs, ys, color=color_contorno, linewidth=1.2)
            except Exception as e:
                print(f"Cluster {cl} alphashape failed: {e}")

        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        #plt.legend(markerscale=5, fontsize=6)
        #plt.savefig(f"cluster_scaled_tile_{nombre_foto}.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.show()

    return df_clustering


def gene_count_csv(df, muestra): 
    '''
    Esta función genera una matriz de conteos a partir del data frame de baysor. 

    Parámentros: 

    - df: Data frame output de la segmentación de baysor. Cada línea correspondía a la información de un tránscrito. 
    - muestra: Un string con el identificativo de la muestra 

    Resultados: una matriz de conteos célula x gen. 

    '''
    # se eliminan los tránscritos con menos de un 0.5 de confianza 
    df_filtered = df[(df['assignment_confidence'] > 0.5) & (df['confidence'] > 0.5) & (df['is_noise'] == False)]
    #de cada célula se cuentan cuantas repeticiones hay de cada gen
    df_gene_count = (
        df_filtered
        .groupby(['cell', 'gene'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    #se eliminan células con menos de 10 tránscritos
    df_gene_count = df_gene_count[df_gene_count.drop(columns=['cell']).sum(axis=1, numeric_only=True) > 10]
    df_gene_count.to_csv(f'C:/Users/ruben/Desktop/RUBEN/BBC_master/3_cuatri/Practicas/trabajo/propio/Analisis/{muestra}/gene_count_{muestra}.csv', index=False)
    
    return df_gene_count

def cell_anotation(df_gene_count, dict_markers): 
    '''
    Asignamos un tipo celular a cada una de las células

    Parámetros: 

    - df_gene_count: Matriz de conteos que devuelve la función gene_count_csv 
    - dict_markers: Diccionario con los tipos celulares de keys y sus genes marcadores en una lista como values

    Resultados: 
    - df_marker_count: data frame que indica el número de tránscritos marcadores de cada tipo, además de una columna indicando el tipo celular
    '''

    #diccionario con la cuenta de los marcadores por tipo celular
    df_marker_count = pd.DataFrame(index=df_gene_count.index)
    for cell_type, genes in dict_markers.items():
    # Filtrar solo los genes que existen en el DataFrame de genes marcadores
        genes_existentes = [g for g in genes if g in df_gene_count.columns]
    
    # Sumar los counts de esos genes para cada célula
        if genes_existentes:
            df_marker_count[cell_type] = df_gene_count[genes_existentes].sum(axis=1)
        else:
            df_marker_count[cell_type] = 0

###anotacion
#En cada tipo celular se usa como criterio que haya mas de 2/3 tránscritos de dicho tipo celular y un porcentaje X 
#de presencia de tránscritos de dicho tipo celular

    df_marker_count['Assigned_Cell_Type'] = 'Unknown'

    #Astrocytes
    df_marker_count.loc[(df_marker_count['Astrocytes']/df_marker_count.sum(axis=1, numeric_only=True) > 0.3) & (df_marker_count['Astrocytes'] > 2),
                        'Assigned_Cell_Type'] = 'Astrocytes'

    #Neurons
    df_marker_count.loc[(df_marker_count['Neurons']/df_marker_count.sum(axis=1, numeric_only=True) > 0.3) & (df_marker_count['Neurons'] > 2),
                        'Assigned_Cell_Type'] = 'Neurons'


    #Melanoma
    df_marker_count.loc[(df_marker_count['Melanoma']/df_marker_count.sum(axis=1, numeric_only=True) > 0.2) & (df_marker_count['Melanoma'] > 2),
                        'Assigned_Cell_Type'] = 'Melanoma'
    
    mask = df_marker_count['Assigned_Cell_Type'].isin(['Astrocytes', 'Neurons', 'Melanoma'])
    df_marker_count.loc[mask, 'Assigned_Cell_Type'] = df_marker_count.loc[mask, ['Astrocytes', 'Neurons', 'Melanoma']].idxmax(axis=1)


    #Mitf también se encuentra en neuronas, entonces si hay Mitf y algún marcador de neuronas, y no hay Dct o Pmel, se trata de una neurona
    df_marker_count.loc[(df_gene_count['Mitf'] > 0) & (df_gene_count['Dct'] < 2) & (df_gene_count['Pmel'] < 2) & (df_marker_count['Neurons'] > 0),
                'Assigned_Cell_Type'] = 'Neurons'

    #Leukocytes
    df_marker_count.loc[(df_marker_count['Leukocytes']/df_marker_count.sum(axis=1, numeric_only=True) > 0.1) & (df_marker_count['Leukocytes'] > 2),
                        'Assigned_Cell_Type'] = 'Leukocytes'

    #T_cells
    df_marker_count.loc[((df_marker_count['T_cells'] + df_marker_count['Leukocytes'])/df_marker_count.sum(axis=1, numeric_only=True) > 0.1) & (df_marker_count['T_cells'] > 2),
                        'Assigned_Cell_Type'] = 'T_cells'

    #B_cells
    df_marker_count.loc[((df_marker_count['B_cells'] + df_marker_count['Leukocytes'])/df_marker_count.sum(axis=1, numeric_only=True) > 0.1) & (df_marker_count['B_cells'] > 2),
                        'Assigned_Cell_Type'] = 'B_cells'

    #Myeloid
    df_marker_count.loc[(df_marker_count['Myeloid']/df_marker_count.sum(axis=1, numeric_only=True) > 0.1) & (df_marker_count['Myeloid'] > 2),
                        'Assigned_Cell_Type'] = 'Myeloid'
 
    #Microglia
    df_marker_count.loc[(df_marker_count['Microglia']/df_marker_count.sum(axis=1, numeric_only=True) > 0.1) & (df_marker_count['Microglia'] > 2),
                        'Assigned_Cell_Type'] = 'Microglia'
    df_marker_count.loc[(df_marker_count['Assigned_Cell_Type'] == 'Myeloid') & (df_marker_count['Microglia'] > 2),
                        'Assigned_Cell_Type'] = 'Microglia' 

    #Dendritic_cells
    df_marker_count.loc[(df_marker_count['Assigned_Cell_Type'] == 'Myeloid') & (df_marker_count['Dendritic_cells'] > 2),
                        'Assigned_Cell_Type'] = 'Dendritic_cells'
    df_marker_count.loc[(df_gene_count['Flt3'] > 0) & (df_gene_count['Itgax'] > 0) & (df_gene_count['Itgae'] > 0),
                        'Assigned_Cell_Type'] = 'Dendritic_cells'
    df_marker_count.loc[(df_gene_count['Flt3'] > 0) & ((df_gene_count['Itgax'] > 1) | (df_gene_count['Itgae'] > 1)) & (df_marker_count['Neurons'] < 10),
                        'Assigned_Cell_Type'] = 'Dendritic_cells' 
    #Si tienen marcadores de neuronas no son dentriticas
    df_marker_count.loc[(df_marker_count['Assigned_Cell_Type'] == 'Myeloid') & (df_gene_count['Cd24a'] > 0) & (df_gene_count['Itgam'] < 1) & (df_gene_count['Adgre1'] < 1) & (df_marker_count['Neurons'] > 0),
                    'Assigned_Cell_Type'] = 'Neurons'
    #Oligodendrocytes
    df_marker_count.loc[(df_marker_count['Oligodendrocytes']/df_marker_count.sum(axis=1, numeric_only=True) > 0.2) & (df_marker_count['Oligodendrocytes'] > 2),
                        'Assigned_Cell_Type'] = 'Oligodendrocytes'

    #Vasculature
    df_marker_count.loc[(df_marker_count['Vasculature']/df_marker_count.sum(axis=1, numeric_only=True) > 0.2) & (df_marker_count['Vasculature'] > 2) & (df_marker_count['Melanoma']/df_marker_count['Vasculature'] < 0.5),
                        'Assigned_Cell_Type'] = 'Vasculature'

    print('Astrocytes', (df_marker_count['Assigned_Cell_Type'] == 'Astrocytes').value_counts().get(True, 0))
    print('Neurons', (df_marker_count['Assigned_Cell_Type'] == 'Neurons').value_counts().get(True, 0))
    print('Melanoma', (df_marker_count['Assigned_Cell_Type'] == 'Melanoma').value_counts().get(True, 0))
    print('Leukocytes', (df_marker_count['Assigned_Cell_Type'] == 'Leukocytes').value_counts().get(True, 0))
    print('B_cells', (df_marker_count['Assigned_Cell_Type'] == 'B_cells').value_counts().get(True, 0))
    print('T_cells', (df_marker_count['Assigned_Cell_Type'] == 'T_cells').value_counts().get(True, 0))
    print('Myeloid', (df_marker_count['Assigned_Cell_Type'] == 'Myeloid').value_counts().get(True, 0))
    print('Microglia', (df_marker_count['Assigned_Cell_Type'] == 'Microglia').value_counts().get(True, 0))
    print('Dendritic_cells', (df_marker_count['Assigned_Cell_Type'] == 'Dendritic_cells').value_counts().get(True, 0))
    print('Oligodendrocytes', (df_marker_count['Assigned_Cell_Type'] == 'Oligodendrocytes').value_counts().get(True, 0))
    print('Vasculature', (df_marker_count['Assigned_Cell_Type'] == 'Vasculature').value_counts().get(True, 0))

    return df_marker_count

def plot_annotation (df_gene_count, df_marker_count, df_baysor_results, gene_color, baysor_polygons, type_color,
                     dict_markers, gene_markers_to_represent = None, cell_types_to_represent = None,  size_fig = 29, 
                     only_markers = True, gene_in = None, cell_in = None):
    '''
    Función para representar las células del tipo que se quiera y los tránscritos de los genes que se quieran

    Parámetros: 
    - df_gene_count: Matriz de conteos que devuelve la función gene_count_csv. Se usa para sacar la lista de todos los genes 
    del experimento. 
    - df_marker_count: data frame que indica el número de tránscritos marcadores de cada tipo, además de una columna indicando el tipo celular
    - df_baysor_results: data frame con la información de los tránscritos. Contiene la localización espacial de cada tránscrito
    - gene_color: diccionario con todos los genes marcadores y un color asociado
    - baysor_polygons: data frame con las coordenadas de los vértices de las células, necesarios para dibujar el contorno de las células
    - dict_markers: Diccionario con los tipos celulares de keys y sus genes marcadores en una lista como values
    - type_color: color del contorno en función del tipo celular
    - gene_in: lista de genes que se quieran plotear
    - size_fig: tamaño de la figura
    - only_markers: True, solo se muestran los genes marcadores, False, se muestran todos los genes (en negro los que no sean marcadores)
    - cell_in: lista del id de las células que se quieran mostrar

    Resultados: 
    - Plot del tejido completo, en el que se pueden mostrar genes y células concretos.
    
    '''


    lista_genes = sorted(df_gene_count.columns)
    fig, ax = plt.subplots(figsize=(size_fig, size_fig))
    if gene_in is None and gene_markers_to_represent is None:
        if not only_markers:
            gene_in = lista_genes
        else:
            gene_in = gene_color.keys()
    elif gene_in is None and gene_markers_to_represent:
        gene_in = []
        for esp in gene_markers_to_represent:
            gene_in.extend(dict_markers[esp])
    elif gene_markers_to_represent:
        for esp in gene_markers_to_represent:
            gene_in.extend(dict_markers[esp])

    for gene in lista_genes:
        if gene in gene_in:  
            df_cell_types_plot = df_baysor_results[df_baysor_results['gene'] == gene]              
            if gene not in gene_color.keys():
                ax.scatter(df_cell_types_plot['x'], df_cell_types_plot['y'], s=0.25, color='black', label=gene, alpha=0.5)
                continue
            ax.scatter(df_cell_types_plot['x'], df_cell_types_plot['y'], s=0.5, color=gene_color[gene], label=gene, alpha=0.5) #únicamente se plotean los 36 genes del color
    
    if cell_in is None and cell_types_to_represent is None:
        cell_in = df_gene_count.index
    elif cell_in is None and cell_types_to_represent:
        cell_in = []
        for esp in cell_types_to_represent:
            cell_in.extend(df_marker_count[df_marker_count['Assigned_Cell_Type'] == esp].index)
    elif cell_types_to_represent:
        for esp in cell_types_to_represent:    
            cell_in.extend(df_marker_count[df_marker_count['Assigned_Cell_Type'] == esp].index)

    for cell_id, group in baysor_polygons.groupby('cell'):
        if cell_id in cell_in:
        # Cerrar el polígono conectando último punto con el primero
            x_coords = list(group['x']) + [group['x'].iloc[0]]
            y_coords = list(group['y']) + [group['y'].iloc[0]]
            color = type_color[df_marker_count.loc[cell_id, 'Assigned_Cell_Type']]
            ax.plot(x_coords, y_coords, linewidth=1, color = color)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    # Añadir leyenda con colores por tipo celular
    #patches = [mpatches.Patch(color=color, label=tipo) for tipo, color in colores_tipo.items()]
    #ax.legend(handles=patches, loc='upper right', fontsize='small', title='Tipos celulares')
    plt.show()
    return

def csv_cytomap (ruta, df_gene_count, df_marker_count, dict_markers, muestra):
    '''
    Crea un arhivo con un formato adecuado para la aplicación cytomap, la cual permite el análisis de muestras con información espacial

    Parámetros:
    - ruta: Ruta de acceso a la carpeta con los archivos que da Baysor con información sobre las células como área, densidad, etc. 
    - df_gene_count: Data frame de pandas con los conteos por gen en cada célula
    - df_marker_count: Data frame que cuenta el número de tránscritos de cada célula que representan a un tipo celular. Se usó para la anotación
    - dict_markers: Diccionario con los genes marcadores de cada tipo celular
    - muestra: Nombre de la muestra

    Resultado: 
    df_cytomap: Data frame con formato para la aplicación Cytomap. Consiste en la matriz de expresión normalizada células x genes 
    y columnas con las coordenadas, el area y el tipo celular al que pertenece cada célula
    '''

    df_cell_stats = pd.read_csv(ruta)
    df_cell_stats = df_cell_stats.drop(columns=['n_transcripts', 'density', 'avg_confidence', 'elongation']).set_index('cell').rename_axis(index=None)
    #normalización
    df_gene_count_norm = np.log1p(df_gene_count.div(df_gene_count.sum(axis=1), axis = 0) * 10000) 

    df_gene_count_norm = df_gene_count_norm.copy()  
    df_gene_count_norm['Cell_type'] = df_marker_count['Assigned_Cell_Type']
    df_cytomap = df_gene_count_norm.join(df_cell_stats, how='left')
    df_cytomap.rename(columns={'x' : 'X', 'y' : 'Y', 'area' : 'Area'}, inplace=True)

    # Crear el directorio si no existe
    directorio = f'C:/Users/ruben/Desktop/RUBEN/BBC_master/3_cuatri/Practicas/trabajo/propio/Analisis/cytomap/{muestra}'
    os.makedirs(directorio, exist_ok=True)
    n = 1
    for tipo in dict_markers.keys():
        df_tipo = df_cytomap[df_cytomap['Cell_type'] == tipo]
        if df_tipo.shape[0] == 0:
            # Crear CSV vacío pero con las columnas correctas
            df_tipo = df_cytomap.iloc[0:0]
            
        df_tipo.to_csv(f'{directorio}/{tipo}.csv', index_label="Cell")

    return df_cytomap

def plot_area(df_cytomap, df_gene_count):
    '''
    Función para plotear la distribución del área de las células y ver así si son similares entre muestras

    Parámetros:
    - df_cytomap: data frame con un formato adecuado para la aplicación cytomap, contiene el área de las células
    - df_gene_count: matriz de expresión

    Resultados:
    Plot de la distribución del área de las células e información sobre el número de tránscritos por célula
    '''
    
    plt.figure()
    plt.hist(df_cytomap['Area'], bins=20, range = (0, 40000))
    plt.xlabel('Area')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución del area de las células ')
    plt.tight_layout()
    plt.show()
    
    print(f'Número de tránscritos total: {df_gene_count.sum().sum()}')
    print(f'Número total de células: {df_gene_count.shape[0]}')
    print(f'Número de tránscritos por célula: {df_gene_count.sum().sum()/df_gene_count.shape[0]}')
    return 

def divide_regions(df_cytomap, radios=[700]):
    """
    Divide el dataframe en regiones concéntricas según la distancia a células de melanoma.

    Parámetros:
    - df_cytomap: Data frame con formato para la aplicación Cytomap. Consiste en la matriz de expresión normalizada células x genes 
    y columnas con las coordenadas, el area y el tipo celular al que pertenece cada célula
    - radios: lista de thresholds en orden ascendente.

    Resultado:
    - regiones: lista de data frames, uno por región, que contiene las células y la información de expresión.  
    
    """
    if isinstance(radios, (int, float)):
        radios = [radios]

    radios = sorted(radios)

    coords_all = df_cytomap[['X', 'Y']].values
    coords_melanoma = df_cytomap[df_cytomap['Cell_type'] == 'Melanoma'][['X', 'Y']].values

    distancias = np.sqrt(
        ((coords_all[:, None, :] - coords_melanoma[None, :, :]) ** 2).sum(axis=2)
    )
    dist_min = distancias.min(axis=1)

    regiones = []

    # Región 1: melanoma + dentro del primer radio
    mask = (df_cytomap['Cell_type'] == 'Melanoma') | (dist_min <= radios[0])
    regiones.append(df_cytomap[mask])
    mask_acumulada = mask

    # Regiones intermedias
    for i in range(1, len(radios)):
        mask = (~mask_acumulada) & (dist_min <= radios[i])
        regiones.append(df_cytomap[mask])
        mask_acumulada = mask_acumulada | mask

    # Última región: todo lo que queda fuera
    regiones.append(df_cytomap[~mask_acumulada])

    for n in range(len(regiones)):
        df = regiones[n]
        print(f'Número de células en la región {n}:', df[df['Cell_type'] != 'Melanoma'].shape[0])

    return regiones

def plot_melanoma_by_region(df_marker_count_list, baysor_polygons, region_colors,
                             melanoma_color="red", cell_type="Melanoma",
                             region_names=None, size_fig=15,
                             show_scatter=True, scatter_size=5, scatter_alpha=0.6):
    """
    Representa los tejidos, con un color para las células de cada región

    Parámetros:
    - df_marker_count_list : lista de DataFrames por región (output de divide_regions)
    - baysor_polygons: data frame con las coordenadas de los vértices de las células, necesarios para dibujar el contorno de las células
    - region_colors: lista con los colores en los que se pinta cada región, en orden creciente de distancia
    - melanoma_color: color en el que se pintan las células de melanoma
    - cell_type: permite seleccionar las células asignadas como melanoma en el data frame
    - region_ names: lista con los nombres en la leyenda de cada región
    - size_fig: tamaño de la figura, para variarla en función del tamaño de los tejidos
    - show_scatter: si True, dibuja un punto en el centroide de cada célula
    - scatter_size: tamaño del centroide
    - scatter_alpha: opacidad del centroide (0 transparente, 1 opaco)

    Resultados: 
    - Representación del tejido por regiones

    """

    fig, ax = plt.subplots(figsize=(size_fig, size_fig))

    # 1. Polígonos de cada región
    for i, df_region in enumerate(df_marker_count_list):
        color = region_colors[i]
        polygons_region = baysor_polygons[baysor_polygons['cell'].isin(df_region.index)]

        for cell_id, group in polygons_region.groupby('cell'):
            x_coords = list(group['x']) + [group['x'].iloc[0]]
            y_coords = list(group['y']) + [group['y'].iloc[0]]
            ax.plot(x_coords, y_coords, linewidth=1, color=color)

    # 2. Scatter centroides por región (no melanoma)
    if show_scatter:
        for i, df_region in enumerate(df_marker_count_list):
            color = region_colors[i]
            df_no_melanoma = df_region[df_region['Cell_type'] != cell_type]
            ax.scatter(
                df_no_melanoma['X'], df_no_melanoma['Y'],
                s=scatter_size, color=color, alpha=scatter_alpha,
                linewidths=0, zorder=2
            )

    # 3. Polígonos melanoma encima (de todas las regiones)
    for df_region in df_marker_count_list:
        cells_melanoma = df_region[df_region['Cell_type'] == cell_type].index
        polygons_melanoma = baysor_polygons[baysor_polygons['cell'].isin(cells_melanoma)]

        for cell_id, group in polygons_melanoma.groupby('cell'):
            x_coords = list(group['x']) + [group['x'].iloc[0]]
            y_coords = list(group['y']) + [group['y'].iloc[0]]
            ax.plot(x_coords, y_coords, linewidth=1, color=melanoma_color)

    # 4. Scatter centroides melanoma encima
    if show_scatter:
        for df_region in df_marker_count_list:
            df_mel = df_region[df_region['Cell_type'] == cell_type]
            ax.scatter(
                df_mel['X'], df_mel['Y'],
                s=scatter_size * 1.5, color=melanoma_color, alpha=scatter_alpha,
                linewidths=0, zorder=3
            )

    # Leyenda
    labels = region_names if region_names else [f"Región {i+1}" for i in range(len(region_colors))]
    handles = [
        plt.Line2D([0], [0], color=region_colors[i], linewidth=2, label=labels[i])
        for i in range(len(region_colors))
    ]
    handles.append(plt.Line2D([0], [0], color=melanoma_color, linewidth=2, label=cell_type))
    ax.legend(handles=handles, loc="upper right")
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def all_sample_df(df_list, cell_type='Astrocytes'):
    """
    Concatena DataFrames de nuestras 3 muestras, seleccionando únicamente las células de un tipo celular concreto
    
    Parámetros:
    - df_list : En nuestro caso, lista con exactamente 9 DataFrames en orden: 
        [BR1_135_2, BR1_135_1, BR1_138, BR3_148, BR3_150_1, BR3_150_2, PBS_172_1, PBS_172_2, PBS_172_3]
    - cell_type : Tipo celular a filtrar
    
    Resultado:
    - Dataframes de expresión génica de distintas muestras con únicamente con un tipo celular
    (df_norm_expression_BR1, df_norm_expression_BR3, df_norm_expression_PBS)
    """
    if len(df_list) != 9:
        raise ValueError("Se necesitan exactamente 9 DataFrames en la lista")
    
    # BR1: primeros 3 DataFrames
    df_norm_expression_BR1 = pd.concat([
        df_list[0][df_list[0]['Cell_type'] == cell_type],
        df_list[1][df_list[1]['Cell_type'] == cell_type],
        df_list[2][df_list[2]['Cell_type'] == cell_type]
    ], axis=0).drop(columns=['X', 'Y', 'Area', 'Cell_type'])
    
    # BR3: siguientes 3 DataFrames
    df_norm_expression_BR3 = pd.concat([
        df_list[3][df_list[3]['Cell_type'] == cell_type],
        df_list[4][df_list[4]['Cell_type'] == cell_type],
        df_list[5][df_list[5]['Cell_type'] == cell_type]
    ], axis=0).drop(columns=['X', 'Y', 'Area', 'Cell_type'])
    
    # PBS: últimos 3 DataFrames
    df_norm_expression_PBS = pd.concat([
        df_list[6][df_list[6]['Cell_type'] == cell_type],
        df_list[7][df_list[7]['Cell_type'] == cell_type],
        df_list[8][df_list[8]['Cell_type'] == cell_type]
    ], axis=0).drop(columns=['X', 'Y', 'Area', 'Cell_type'])
    
    return df_norm_expression_BR1, df_norm_expression_BR3, df_norm_expression_PBS


def calculate_deg(df_group1, df_group2, test='ttest'):
    """
    Calcula log fold change y p-valores entre dos grupos

    Parámetros: 
    - df_group1, df_group2: DataFrames con genes en columnas, células en filas
    - test: 'ttest' o 'mannwhitneyu'

    Resultado:
    - df_results: Data frame en el que cada fila es un gen. Se incluye el pvalue y el logFC entre dos grupos, 
    así como la media de cada uno de los grupos 
    """
    results = []
    
    for gene in df_group1.columns:
        # Expresión media de cada grupo
        mean_group1 = df_group1[gene].mean()
        mean_group2 = df_group2[gene].mean()
        
        # log Fold Change (la información ya está como logaritmo)
        logfc = mean_group1 - mean_group2
        
        # Test estadístico
        if test == 'ttest':
            stat, pval = ttest_ind(df_group1[gene], df_group2[gene])
        else:
            stat, pval = mannwhitneyu(df_group1[gene], df_group2[gene])
        
        results.append({
            'gene': gene,
            'logFC': logfc,
            'pvalue': pval,
            'mean_group1': mean_group1,
            'mean_group2': mean_group2
        })
    
    df_results = pd.DataFrame(results)
    
    # Ajuste de p-valores (FDR - Benjamini-Hochberg)
    df_results['padj'] = multipletests(df_results['pvalue'], method='fdr_bh')[1]
    
    # -log10(p-value) para volcano plot
    df_results['-log10pval'] = -np.log10(df_results['pvalue'])
    df_results['-log10padj'] = -np.log10(df_results['padj'])
    
    return df_results


def multi_comparison_heatmap(deg_br1_vs_pbs, deg_br3_vs_pbs, deg_br1_vs_br3, cell_type, top_n=15, padj_threshold=0.05):
    """
    Heatmap mostrando logFC de múltiples comparaciones

    Parámetros: 
    - deg_br1_vs_pbs, deg_br3_vs_pbs, deg_br1_vs_br3: como tenemos 3 condiciones, ploteamos las 3 comparaciones posibles.
    Es el output de la función calculate_deg
    - cell_type: tipo celular de la comparación. Es para el título del heatmap
    - top_n: número de genes de cada comparación que se quieren tomar. 
    Se plotearán más genes si no coinciden los top_n en todas las comparaciones
    - padj_threshold: umbral a partir del cual un resultado se considera significativo y se marca con un asterisco. 

    Resultado:
    - fig: heatmap de expresión diferencial
    """
    # Top genes de cada comparación
    top_genes_1 = deg_br1_vs_pbs.nsmallest(top_n, 'padj')['gene'].tolist()
    top_genes_2 = deg_br3_vs_pbs.nsmallest(top_n, 'padj')['gene'].tolist()
    top_genes_3 = deg_br1_vs_br3.nsmallest(top_n, 'padj')['gene'].tolist()
    
    # Unir todos los genes únicos
    all_top_genes = list(set(top_genes_1 + top_genes_2 + top_genes_3))
    
    # Crear DataFrame con logFC de cada comparación
    df_heatmap = pd.DataFrame({
        'BR1 vs PBS': deg_br1_vs_pbs.set_index('gene').loc[all_top_genes, 'logFC'],
        'BR3 vs PBS': deg_br3_vs_pbs.set_index('gene').loc[all_top_genes, 'logFC'],
        'BR1 vs BR3': deg_br1_vs_br3.set_index('gene').loc[all_top_genes, 'logFC']
    }).T
    df_padj = pd.DataFrame({
        'BR1 vs PBS': deg_br1_vs_pbs.set_index('gene').loc[all_top_genes, 'padj'],
        'BR3 vs PBS': deg_br3_vs_pbs.set_index('gene').loc[all_top_genes, 'padj'],
        'BR1 vs BR3': deg_br1_vs_br3.set_index('gene').loc[all_top_genes, 'padj']
    }).T

    annot = df_padj.applymap(lambda x: '*' if x < padj_threshold else '')

    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(df_heatmap, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'log Fold Change'},
                linewidths=0.5, ax=ax,
                vmin=-3, vmax=3, annot=annot, fmt='s', annot_kws={'fontsize': 12, 'weight': 'bold'})
    ax.set_title(f'{cell_type}', fontsize=14)
    ax.set_xlabel('Genes', fontsize=12)
    ax.set_ylabel('Comparisons', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    return fig



def all_sample_division_df(df_list, cell_type='Astrocytes', radio=700):
    """
    Divide cada muestra en regiones y concatena por tipo (BR1/BR3) y región
    
    Parámetros:
    
    - df_list : Lista con exactamente 6 DataFrames en orden: 
        [BR1_135_2, BR1_135_1, BR1_138, BR3_148, BR3_150_1, BR3_150_2]
    - cell_type : Tipo celular a filtrar 
    - radio : Radio en unidades para definir las regiones

    Resultado:
    - (df_BR1_region1, df_BR1_region2, df_BR3_region1, df_BR3_region2): 
    Conjunto de Data Frames con la expresión de cada región de cada muestra
    """
    if len(df_list) != 6:
        raise ValueError("Se necesitan exactamente 9 DataFrames en la lista")
    

    # Listas para almacenar regiones de cada muestra
    br1_region1_list = []
    br1_region2_list = []
    br3_region1_list = []
    br3_region2_list = []
    
    # Procesar BR1 (muestras 0, 1, 2)
    for i in range(3):
        r1, r2 = divide_regions(df_list[i], radio)
        br1_region1_list.append(r1[r1['Cell_type'] == cell_type])
        br1_region2_list.append(r2[r2['Cell_type'] == cell_type])
    
    # Procesar BR3 (muestras 3, 4, 5)
    for i in range(3, 6):
        r1, r2 = divide_regions(df_list[i], radio)
        br3_region1_list.append(r1[r1['Cell_type'] == cell_type])
        br3_region2_list.append(r2[r2['Cell_type'] == cell_type])
    
    # Concatenar y limpiar
    df_BR1_region1 = pd.concat(br1_region1_list, axis=0).drop(columns=['X', 'Y', 'Area', 'Cell_type'])
    df_BR1_region2 = pd.concat(br1_region2_list, axis=0).drop(columns=['X', 'Y', 'Area', 'Cell_type'])
    df_BR3_region1 = pd.concat(br3_region1_list, axis=0).drop(columns=['X', 'Y', 'Area', 'Cell_type'])
    df_BR3_region2 = pd.concat(br3_region2_list, axis=0).drop(columns=['X', 'Y', 'Area', 'Cell_type'])
    
    print(f"BR1 Región 1 ({cell_type}): {len(df_BR1_region1)} células")
    print(f"BR1 Región 2 ({cell_type}): {len(df_BR1_region2)} células")
    print(f"BR3 Región 1 ({cell_type}): {len(df_BR3_region1)} células")
    print(f"BR3 Región 2 ({cell_type}): {len(df_BR3_region2)} células")
    
    return df_BR1_region1, df_BR1_region2, df_BR3_region1, df_BR3_region2


def region_comparison_heatmap(deg_list, comparison_names, cell_type,
                              top_n=15, padj_threshold=0.05):
    """
    Heatmap mostrando logFC de múltiples comparaciones

    Parámetros: 
    - deg_list: lista de data frames que contienen el logfc y p valor de los genes entre varias comparaciones
    - comparison_names: lista de los nombres de las comparaciones que se desea representar
    - cell_type: tipo celular que se ha comparado, es para el título de la figura
    - top_n: número de genes de cada comparación que se quieren tomar. 
    Se plotearán más genes si no coinciden los top_n en todas las comparaciones
    - padj_threshold: umbral a partir del cual un resultado se considera significativo y se marca con un asterisco.

    Resultado:
    - fig: heatmap de expresión diferencial
    - genes_heatmap: lista de genes que se muestran en el heatmap  
    """

    if len(deg_list) != len(comparison_names):
        raise ValueError("deg_list y comparison_names deben tener la misma longitud")

    # Recopilar top genes de todas las comparaciones (manteniendo orden)
    all_top_genes = []
    for deg in deg_list:
        top_genes = deg.nsmallest(top_n, 'padj')['gene'].tolist()
        for g in top_genes:
            if g not in all_top_genes:
                all_top_genes.append(g)

    genes_heatmap = all_top_genes.copy()

    # Crear DataFrames de logFC y padj
    fc_dict = {}
    padj_dict = {}

    for deg, name in zip(deg_list, comparison_names):
        deg_indexed = deg.set_index('gene')

        fc_dict[name] = deg_indexed.reindex(genes_heatmap)['logFC']
        padj_dict[name] = deg_indexed.reindex(genes_heatmap)['padj']

    df_heatmap = pd.DataFrame(fc_dict).T
    df_padj = pd.DataFrame(padj_dict).T

    # Anotaciones de significancia
    annot = df_padj.applymap(lambda x: '*' if x < padj_threshold else '')

    # Plot
    fig, ax = plt.subplots(figsize=(16, max(6, len(comparison_names) * 0.8)))
    sns.heatmap(
        df_heatmap,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'log Fold Change'},
        linewidths=0.5,
        ax=ax,
        vmin=-3, vmax=3,
        annot=annot, fmt='s',
        annot_kws={'fontsize': 12, 'weight': 'bold'}
    )

    ax.set_title(f'{cell_type} - Differential Expression', fontsize=14, weight='bold')
    ax.set_xlabel('Genes', fontsize=12)
    ax.set_ylabel('Comparisons', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()

    print(f"✓ Heatmap generado con {len(genes_heatmap)} genes únicos")
    print(f"✓ Comparaciones: {len(comparison_names)}")

    return fig, genes_heatmap

