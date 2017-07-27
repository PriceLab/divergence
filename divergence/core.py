import pandas
import logging


class Divergence:
    """Class for creating divergence matrices.

    Args:
        training_data (pandas.DataFrame): an analytes(columns) by individuals
            (index) dataframe to train the analyte ranges on
        lower (float): lower percentile bound
        upper (float): upper percentile bound
        quantize (bool): Whether to convert the given training data into
            quantiles
    """
    def __init__(self, training_data, lower=.025, upper=.975, quantize=False):
        self._ranges = None
        self._l = lower
        self._u = upper
        if quantize:
            training_data = self.quantize(training_data)
        self._training_data = training_data

    @property
    def ranges(self):
        """dict: a dictionary key->analyte name, value-> tuple(lower bound,
                upper bound)"""
        if self._ranges is None:
            self._ranges = self.get_ranges(self._training_data, self._l, self._u)
        return self._ranges

    def get_ranges(self, training_data, lower, upper):
        """Generates dictionary containing the upper and lower bounds for
        divergence given the training data.

        Args:
            training_data (pandas.DataFrame): an analytes(columns) by
                individuals (index) dataframe to train the analyte ranges on,
                that has already been quantized
            lower (float): lower percentile bound
            upper (float): upper percentile bound
        """
        ranges = {}
        for c in training_data.columns:
            a = training_data[c]
            a = a.dropna().tolist()
            a.sort()
            l = int(len(a)*lower)
            u = int(len(a)*upper)
            if u < len(a) - 1:
                while a[u] == a[u+1]:
                    u -= 1
            else:
                u = len(a) - 1
                logging.debug("%s does not have enough values to choose a range at the specified level" % c)
            if l > 0:
                while a[l] == a[l-1]:
                    l += 1
            else:
                l = 0
            ranges[c] = (a[l], a[u])
        return ranges

    def quantize(self, df):
        """
        Quantizes a dataframe.

        Args:
            df (pandas.Dataframe): an analytes(columns) by
                individuals (index) dataframe

        Returns:
            pandas.DataFrame: the quantized transformation of `df`
        """
        return df.rank(pct=True, axis=1)

    def transform(self, df, quantize=False):
        """Creates a divergence matrix based on provided training data.

        Args:
            df (pandas.DataFrame): an analytes(columns) by
                individuals (index) dataframe to transform into a divergence
                matrix using trained ranges.
            quantize (bool): Whether to convert the given training data into
                quantiles

        Returns:
            pandas.DataFrame: the divergence transformation of `df`
         """
        if quantize:
            df = self.quantize(df)
        temp = {}
        for c in df.columns:
            l, u = self.ranges[c]
            a = df[c].copy()
            zeros = ((a >= l) & (a <= u))
            negs = (a < l)
            pos = (a > u)
            a[zeros] = 0.0
            a[negs] = -1.0
            a[pos] = 1.0
            temp[c] = a
        return pandas.DataFrame(temp)


def negative_only(div):
    """Get divergence matrix containing only negatively divergent values

    Args: 
        div (pandas.DataFrame): a divergence matrix
    
    Returns:
        pandas.DataFrame:
            The given divergence matrix with positively divergent values set
            to zero
    """
    t = div.copy()
    t[t==1.0] = 0.0
    return t


def positive_only(div):
    """Get divergence matrix containing only positively divergent values

    Args: 
        div (pandas.DataFrame): a divergence matrix
    
    Returns:
        pandas.DataFrame:
            The given divergence matrix with negatively divergent values set
            to zero
    """
    t = div.copy()
    t[t == -1.0] = 0.0
    return t


def calculate_probabilities(div, over='absolute'):
    """Calculates the probability of divergence for each analyte.

    Args:
        div (pandas.DataFrame): a divergence matrix
        over (str): The conditions over which to generate the probabilities,
            i.e. probability of absolute value, probability of negative divergence,
            probability of positive divergence.One of [`absolute`, `positive`, `negative`]

    Returns:
        pandas.DataFrame:
            The probability of divergence for each variable.
    """
    sdiv = div.copy()
    if over == 'absolute':
        sdiv = sdiv.abs()
    elif over == 'positive':
        sdiv = positive_only(sdiv)
    elif over == 'negative':
        sdiv = negative_only(sdiv).abs()
    else:
        raise Exception("Bad over value")
    return sdiv.sum()/len(sdiv.index)
