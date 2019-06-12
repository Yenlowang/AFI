
import numpy as np
from scipy.stats import norm        # Para disponer de los momentos de la distr. normal

N = norm.cdf                        # Definimos función N como f. distr normal

def BlackScholes(Spot, Strike, TTM, rate, dividends, Vol, IsCall):
    if TTM > 0:

        d1 = (np.log(Spot / Strike) + (rate - dividends + Vol * Vol / 2) * TTM) / (Vol * np.sqrt(TTM))
        d2 = (np.log(Spot / Strike) + (rate - dividends - Vol * Vol / 2) * TTM) / (Vol * np.sqrt(TTM))

        if IsCall:

            return Spot *np.exp(-dividends * TTM)* N(d1) - Strike * np.exp(-rate * TTM) * N(d2)

        else:

            return -Spot *np.exp(-dividends * TTM)* N(-d1) + Strike * np.exp(-rate * TTM) * N(-d2)

    else:

        if IsCall:

            return np.maximum(Spot - Strike, 0)

        else:

            return np.maximum(-Spot + Strike, 0)



