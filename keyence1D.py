# Autheur : TAR
# Date : 13/05/2021
#TODO : ajouter pvalue pour le test de Henry
#TODO : ajouter la densité à l'histogramme pour la répétabilité
#TODO : Ajouter le tmin et tmax pour chaque fonction
#TODO : importer le fichier de settings pour lire la base de temps et précision
#TODO : ajouter la méthode output.file pour avoir un tableau résumé
#TODO : traiter les données #OVER pour les fonction fft et dynamic
#TODO : ajouter la posibilité d'enregister les figures : OK pour repetabilité uniquement


def repetabilityMeasure(Fichier, pngSave=False, IT = 0):
    """Lecture d'un fichier csv du controleur de télémètre laser Keyence
    pour mesure la répétabilité de positionnement

        Parameters
        ----------
        Fichier : "str"
            The file location of the spreadsheet

        Returns
        -------
        plot
            Affichage du nuage de points
            Affichage de l'histogramme
            Affichage du box plot
            Affichage de la droite de Henry
        """


    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import pylab
    import scipy.stats as stats
    from scipy.stats import shapiro



    # Lecture du fichier csv avec les raw datas
    df = pd.read_csv(Fichier, header = 3)

    # On renomme la première colonne du df avec le nom position
    v = df.columns[0]
    df = df.rename(columns={v: 'position'})

    # supression des valeurs out of range '#-OVER'
    df = df[df.position != '#-OVER']

    # passage des valeurs position en float
    df['position'] = df['position'].astype('float')

    # Filtrage des données uniquement dans l'intervalle [-100:100]
    df = df.loc[df['position'] > -100]
    df = df.loc[df['position'] < 100]

    # Reset de l'index
    df = df.reset_index()

    # déclaration de la zone de graphs
    fig = plt.figure(figsize=(14, 9))

    sns.set_theme()
    sns.set_style("white")

    # Matrice graphique 2 x 2 et position 1
    plt.subplot(221)
    plt.scatter(df.index, df.position, c='red')
    plt.xlabel('Nombre de points')
    plt.ylabel('Position (mm)')
    plt.title('Position')
    plt.grid(True, linestyle = '-')

    plt.subplot(222)
    plt.boxplot([df.position])
    plt.xlabel('position')
    plt.ylabel('(mm)')
    plt.title('Box plot des position')

    statData = pd.DataFrame(df['position'].describe())
    plt.table(cellText=statData.values.round(2),
              rowLabels=statData.index,
              colLabels = statData.columns,
              cellLoc = 'right',
              rowLoc = 'center',
              loc='right', bbox=[.7,.05,.3,.8])

    plt.subplot(223)
    plt.hist(df.position, bins=12, rwidth=0.8)
    plt.xlabel('Classes')
    plt.ylabel('Nombre de classes (mm)')
    #sns.displot(df.position, bins=10, kde=True, rug=True, color='red');
    plt.title('Histogramme des position')


    plt.subplot(224)
    measurements = df.position
    stats.probplot(measurements, dist="norm", plot=pylab)
    plt.xlabel('Quantiles théoriques')
    plt.ylabel('Quantiles observés')
    plt.title('qq-plot - Droite de Henry')
    plt.grid(True, linestyle = '--')

    plt.tight_layout()

# ----------------------------------------------------------------------------------------------------------------------

    # normality test
    stat, p = shapiro(df['position'])
    print(Fichier + ' as Statistics=%.3f, p=%.5f' % (stat, p))
    alpha = 0.005
    if p > alpha:
        # plt.figtext(x=0.1, y=0.02, s='Sample looks Gaussian (fail to reject H0) with Statistics=%.3f, p=%.5f' % (stat, p))
        cp = IT / 6 * df['position'].std()
        plt.figtext(x=0.3, y=0.003,
                    s='Sample looks Gaussian (fail to reject H0) with Statistics=%.3f, p=%.5f' % (stat, p))
        print('Cp =%.3f ' %cp)
    else:
        # plt.figtext(x=0.1, y=0.02, s='Sample does not look Gaussian (reject H0) with Statistics=%.3f, p=%.5f' % (stat, p))
        plt.figtext(x=0.3, y=0.003,
                    s='Sample does not look Gaussian (reject H0) with Statistics=%.3f, p=%.5f' % (stat, p))

# ----------------------------------------------------------------------------------------------------------------------

    fig.suptitle(Fichier, fontsize=16)
    if pngSave:
        fileName = Fichier.split('.')[0]
        return plt.savefig(fileName)

    return plt.show()

def fftMeasure(Fichier):
    """Lecture d'un fichier csv du controleur de télémètre laser Keyence
    pour mesurer les fréqences d'oscillations

        Parameters
        ----------
        Fichier : "str"
            The file location of the spreadsheet

        Returns
        -------
        plot
            Affichage de la transformé de Fourier
            Affichage de la mesure brute
        """

    import numpy as np
    from matplotlib import pyplot as plt
    import pandas as pd
    from scipy.fft import fft, fftfreq

    #lecture du fichier csv via pd avec comme en tête la ligne numéro 3
    df = pd.read_csv(Fichier, sep =';', header = 3)

    # Conversion de la colonne index en array
    # les mesures ont été realisé pour un échantillonage de 1ms par point
    t = df.index.to_numpy()
    t = t / 1000

    # Conversion de la df en array
    df_array = df.to_numpy()

    # Ajout de la valeur t en seconde
    t = t.reshape(-1, 1)

    # Création d'un array avec les valeurs x(t) et y(position en mm)
    array = np.concatenate((t, df_array), axis=1)

    x = array[0 : 20000, 0]
    y = array[0 : 20000, 1]

    # Number of sample points
    N = 10000
    # sample spacing
    T = 1.0 / 10000.0


    yf = fft(x)
    xf = fftfreq(N, T)[:N//2]

    # Calcul de la fft pour y et x
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]

    #affichage fft
    fig = plt.figure(figsize=(14, 10))


    plt.subplot(211)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.title(Fichier)

    # affichage signal
    plt.subplot(212)
    plt.plot(x, y)


    return plt.show()

def fftMeasureTwoplots(Fichier1, Fichier2):
    """Lecture de deux fichiers csv du controleur de télémètre laser Keyence
    pour mesurer les fréqences d'oscillations pour comparaison

        Parameters
        ----------
        Fichier1 : "str"
            The file location of the spreadsheet
        Fichier2 : "str"
            The file location of the spredsheet

        Returns
        -------
        plot
            Affichage de la transformé de Fourier du Fichier1 et Fichier2
            Affichage de la mesure brute du Fichier1 et Fichier2
        """

    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.fft import fft, fftfreq

    #lecture du fichier csv via pd avec comme en tête la ligne numéro 3
    df1 = pd.read_csv(Fichier1, sep=';', header=3)
    df2 = pd.read_csv(Fichier2, sep=';', header=3)

    # Conversion de la colonne index en array
    # les mesures ont été realisé pour un échantillonage de 1ms par point
    t1 = df1.index.to_numpy()
    t2 = df2.index.to_numpy()
    t1 = t1 / 1000
    t2 = t2 / 1000

    # Conversion de la df en array
    df_array1 = df1.to_numpy()
    df_array2 = df2.to_numpy()

    # Ajout de la valeur t en seconde
    t1 = t1.reshape(-1, 1)
    t2 = t2.reshape(-1, 1)

    # Création d'un array avec les valeurs x(t) et y(position en mm)
    array1 = np.concatenate((t1, df_array1), axis=1)
    array2 = np.concatenate((t2, df_array2), axis=1)

    x1 = array1[10000 : 20000, 0]
    y1 = array1[10000 : 20000, 1]

    x2 = array2[10000 : 20000, 0]
    y2 = array2[10000 : 20000, 1]

    # Number of sample points
    N = 10000
    # sample spacing
    T = 1.0 / 1000.0


    yf1 = fft(x1)
    yf2 = fft(x2)
    xf1 = fftfreq(N, T)[:N//2]
    xf2 = fftfreq(N, T)[:N//2]


    # Calcul de la fft pour y et x
    yf1 = fft(y1)
    yf2 = fft(y2)
    xf1 = fftfreq(N, T)[:N//2]
    xf2 = fftfreq(N, T)[:N // 2]

    sns.set_theme()


    #affichage fft
    fig = plt.figure(figsize=(14, 10))

    plt.subplot(211)
    plt.plot(xf1, 2.0/N * np.abs(yf1[0:N//2]), xf2, 2.0/N * np.abs(yf2[0:N//2]), label= 'test')
    plt.grid(True, linestyle = '--')
    plt.title(Fichier1)

    # affichage signal
    plt.subplot(212)
    plt.plot(x1, y1, x2, y2)
    plt.grid(True, linestyle='--')


    return plt.show()

def dynamicMeasure(Fichier):
    """Lecture d'un fichiers csv du controleur de télémètre laser Keyence
        pour mesurer la vitesse d'un element

            Parameters
            ----------
            Fichier1 : "str"
                The file location of the spreadsheet

            Returns
            -------
            plot
                Affichage de la mesure
                Affichage de la derivée de la mesure
            """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Lecture du fichier csv avec les raw datas
    df = pd.read_csv(Fichier, header = 3)

    # On renomme la première colonne du df avec le nom position
    v = df.columns[0]
    df = df.rename(columns={v: 'position'})

    # supression des valeurs out of range '#-OVER'
    df = df[df.position != '#-OVER']

    # passage des valeurs position en float
    df['position'] = df['position'].astype('float')

    df_diff = df.diff()
    df_diff = df_diff.rolling(100, win_type='triang').sum()

    # déclaration de la zone de graphs
    fig = plt.figure(figsize=(14, 10))

    sns.set_theme()

    # graphique position en fonction du temps
    plt.subplot(211)
    plt.plot(df.index, df.position, 'r')
    plt.xlabel('Nombre de points')
    plt.ylabel('temps (ms)')
    plt.title('Position en fonction du temps')
    plt.grid(True, linestyle = '--')

    # graphique vitesse en fonction du temps
    plt.subplot(212)
    plt.plot(df.index, df_diff.position, 'b')
    plt.xlabel('vitesse (mm/ms)')
    plt.ylabel('temps (ms)')
    plt.title('Position en fonction du temps')
    plt.grid(True, linestyle = '--')

    fig.suptitle(Fichier, fontsize=16)


    return plt.show()

if __name__ == '__main__':
    repetabilityMeasure()
    fftMeasure()
    fftMeasureTwoplots()
    dynamicMeasure()