import pickle            as pkl
import numpy             as np

def load_data(the_dataset,the_representation):
    
    """
    Load human dissimilarity ratings and the audio sample representations.
    
    Args:
    
        the_dataset: string containing the name of the dataset among
        - 'Grey1977'
        - 'Grey1978'
        - 'Iverson1993_Whole'
        - 'Iverson1993_Onset'
        - 'Iverson1993_Remainder'
        - 'McAdams1995'
        - 'Lakatos2000_Harm'
        - 'Lakatos2000_Perc'
        - 'Lakatos2000_Comb'
        - 'Barthet2010'
        - 'Patil2012_A3'
        - 'Patil2012_DX4'
        - 'Patil2012_GD4'
        - 'Siedenburg2016_e2set1'
        - 'Siedenburg2016_e2set2'
        - 'Siedenburg2016_e2set3'
        - 'Siedenburg2016_e3'

        the_representation: string containing the name of the representation among
        - 'strf'
        - 'stft'
        - 'spectrum' (cochlea in the companion paper)
        - 'scattering'
        - 'clap'
        - 'encodec'
        - 'mert'
        - 'mertcat'
    
    
    Returns:
    
        r: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        D: human dissimilarity ratings stored in a l x l matrix
        
        d: human dissimilarity ratings stored in a l x (l-1)/2 vector
    """
    
    with open('data/'+the_representation+'/'+the_dataset+'.pkl','rb') as Sp:
    
        data = pkl.load(Sp)
        
    # representation
    r = data['representations']
    
    # dissimilarity matrix
    D = data['dissimilarities']
    
    # dissimilarity vector
    P = np.size(D,0)
    d = np.zeros(P*(P-1)//2)
    l = 0
    for i in range(P):
        for j in range(i+1,P):
            d[l] = D[i,j]
            l += 1
    
    return r,D,d
    
    