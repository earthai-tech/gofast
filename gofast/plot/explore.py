# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
`explore` is a set of base plots for visualization, data exploratory and 
analyses. E-E-Q Plots encompass the Exploratory plots
( :class:`~gofast.plot.QuestPlotter`) and Quick analyses 
(:class:`~gofast.plot.EasyPlotter`) visualization. 
"""
from __future__ import annotations 

import re 
import copy
import warnings
import numpy as np 
import  matplotlib.pyplot  as plt
import matplotlib.ticker as mticker
import pandas as pd 
from pandas.plotting import radviz , parallel_coordinates
import seaborn as sns 

from .._gofastlog import gofastlog 
from ..api.docstring import  DocstringComponents,_core_docs,_baseplot_params
from ..api.property import BasePlot
from ..api.types import _F, Any, List,Dict,Optional,ArrayLike, DataFrame, Series  
from ..exceptions import PlotError, FeatureError, NotFittedError

from ..tools._dependency import import_optional_dependency 
from ..tools.coreutils import _assert_all_types , _isin,  repr_callable_obj 
from ..tools.coreutils import smart_strobj_recognition, smart_format, reshape
from ..tools.coreutils import  shrunkformat, exist_features  
from ..tools.baseutils import is_readable, select_features, extract_target
from ..tools.baseutils import generate_placeholders
from ..tools.validator import check_X_y 
try: 
    import missingno as msno 
except : pass 
try : 
    from yellowbrick.features import (
        JointPlotVisualizer, 
        Rank2D, 
        RadViz, 
        ParallelCoordinates,
        )
except: pass 

 
_logger=gofastlog.get_gofast_logger(__name__)

#+++++++++++++++++++++++ add seaborn docs +++++++++++++++++++++++++++++++++++++ 
_sns_params = dict( 
    sns_orient="""
sns_orient: 'v' | 'h', optional
    Orientation of the plot (vertical or horizontal). This is usually inferred 
    based on the type of the input variables, but it can be used to resolve 
    ambiguity when both x and y are numeric or when plotting wide-form data. 
    *default* is ``v`` which refer to 'vertical'  
    """, 
    sns_style="""
sns_style: dict, or one of {darkgrid, whitegrid, dark, white, ticks}
    A dictionary of parameters or the name of a preconfigured style.
    """, 
    sns_palette="""
sns_palette: seaborn color paltte | matplotlib colormap | hls | husl
    Palette definition. Should be something color_palette() can process. the 
    palette  generates the point with different colors
    """, 
    sns_height="""
sns_height:float, 
    Proportion of axes extent covered by each rug element. Can be negative.
    *default* is ``4.``
    """, 
    sns_aspect="""
sns_aspect: scalar (float, int)
    Aspect ratio of each facet, so that aspect * height gives the width of 
    each facet in inches. *default* is ``.7``
    """, 
    )
_qkp_params = dict (  
    classes ="""
classes: list of int | float, [categorized classes] 
    list of the categorial values encoded to numerical. For instance, for
    `flow` data analysis in the Bagoue dataset, the `classes` could be 
    ``[0., 1., 3.]`` which means:: 
        
    * 0 m3/h  --> FR0
    * > 0 to 1 m3/h --> FR1
    * > 1 to 3 m3/h --> FR2
    * > 3 m3/h  --> FR3    
    """, 
    mapflow ="""   
mapflow: bool, 
    Is refer to the flow rate prediction using DC-resistivity features and 
    work when the `target_name` is set to ``flow``. If set to True, value 
    in the target columns should map to categorical values. Commonly the 
    flow rate values are given as a trend of numerical values. For a 
    classification purpose, flow rate must be converted to categorical 
    values which are mainly refered to the type of types of hydraulic. 
    Mostly the type of hydraulic system is in turn tided to the number of 
    the living population in a specific area. For instance, flow classes 
    can be ranged as follow: 

    * FR = 0 is for dry boreholes
    * 0 < FR ≤ 3m3/h for village hydraulic (≤2000 inhabitants)
    * 3 < FR ≤ 6m3/h  for improved village hydraulic(>2000-20 000inhbts) 
    * 6 <FR ≤ 10m3/h for urban hydraulic (>200 000 inhabitants). 
    
    Note that the flow range from `mapflow` is not exhaustive and can be 
    modified according to the type of hydraulic required on the project.   
    """
)
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    base=DocstringComponents(_baseplot_params), 
    sns = DocstringComponents(_sns_params), 
    qdoc= DocstringComponents(_qkp_params)
    )
#++++++++++++++++++++++++++++++++++ end +++++++++++++++++++++++++++++++++++++++

class QuestPlotter (BasePlot): 
    """
    Exploratory plot for data analysis 
    
    `QuestPlotter` is a shadow class. Explore data is needed to create a model since 
    it gives a feel for the data and also at great excuses to meet and discuss 
    issues with business units that controls the data. `QuestPlotter` methods i.e. 
    return an instancied object that inherits from :class:`gofast.property.Baseplots`
    ABC (Abstract Base Class) for visualization.
        
    Parameters 
    -----------
    {params.base.savefig}
    {params.base.fig_dpi}
    {params.base.fig_num}
    {params.base.fig_size}
    {params.base.fig_orientation}
    {params.base.fig_title}
    {params.base.fs}
    {params.base.ls}
    {params.base.lc}
    {params.base.lw}
    {params.base.alpha}
    {params.base.font_weight}
    {params.base.font_style}
    {params.base.font_size}
    {params.base.ms}
    {params.base.marker}
    {params.base.marker_facecolor}
    {params.base.marker_edgecolor}
    {params.base.marker_edgewidth}
    {params.base.xminorticks}
    {params.base.yminorticks}
    {params.base.bins}
    {params.base.xlim}
    {params.base.ylim}
    {params.base.xlabel}
    {params.base.ylabel}
    {params.base.rotate_xlabel}
    {params.base.rotate_ylabel}
    {params.base.leg_kws}
    {params.base.plt_kws}
    {params.base.glc}
    {params.base.glw}
    {params.base.galpha}
    {params.base.gaxis}
    {params.base.gwhich}
    {params.base.tp_axis}
    {params.base.tp_labelsize}
    {params.base.tp_bottom}
    {params.base.tp_labelbottom}
    {params.base.tp_labeltop}
    {params.base.cb_orientation}
    {params.base.cb_aspect}
    {params.base.cb_shrink}
    {params.base.cb_pad}
    {params.base.cb_anchor}
    {params.base.cb_panchor}
    {params.base.cb_label}
    {params.base.cb_spacing}
    {params.base.cb_drawedges} 
    {params.sns.sns_orient}
    {params.sns.sns_style}
    {params.sns.sns_palette}
    {params.sns.sns_height}
    {params.sns.sns_aspect}
    
    Returns
    --------
    {returns.self}
    
    Examples
    ---------
    >>> import pandas as pd 
    >>> from gofast.plot.explore import QuestPlotter
    >>> data = pd.read_csv ('data/geodata/main.bagciv.data.csv' ) 
    >>> QuestPlotter(fig_size = (12, 4)).fit(data).missing(kind ='corr')
    ... <gofast.plot.explore .QuestPlotter at 0x21162a975e0>
    """.format(
        params=_param_docs,
        returns= _core_docs["returns"],
    )    
    msg = ("{expobj.__class__.__name__} instance is not"
           " fitted yet. Call 'fit' with appropriate"
           " arguments before using this method."
           )
    
    def __init__(
        self, 
        target_name:str = None, 
        inplace:bool = False, 
        **kws
        ):
        super().__init__(**kws)
        
        self.target_name= target_name
        self.inplace= inplace 
        self.data= None 
        self.target_= None
        self.y_= None 
        self.xname_=None 
        self.yname_=None 
        

    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `QuestPlotter` is not fitted yet."""
        if self.data is None: 
            raise NotFittedError(self.msg.format(
                expobj=self)
            )
        return 1 
     
    def save (self, fig): 
        """ savefigure if figure properties are given. """
        if self.savefig is not None: 
            fig.savefig (self.savefig, dpi = self.fig_dpi , 
                         bbox_inches = 'tight'
                         )
        plt.show() if self.savefig is None else plt.close () 
        
    def fit(self, data: str |DataFrame,  **fit_params )->'QuestPlotter': 
        """ Fit data and populate the arguments for plotting purposes. 
        
        There is no conventional procedure for checking if a method is fitted. 
        However, an class that is not fitted should raise 
        :class:`exceptions.NotFittedError` when a method is called.
        
        Parameters
        ------------
        data: Filepath or Dataframe or shape (M, N) from 
            :class:`pandas.DataFrame`. Dataframe containing samples M  
            and features N

        fit_params: dict 
            Additional keywords arguments for reading the data is given as 
            a path-like object passed from 
            :func:gofast.tools.coreutils._is_readable`
           
        Return
        -------
        ``self``: `Plot` instance 
            returns ``self`` for easy method chaining.
             
        """
        if data is not None: 
            self.data = is_readable(data, **fit_params)
        if self.target_name is not None:
            self.target_, self.data  = extract_target(
                self.data , self.target_name, self.inplace ) 
            self.y_ = reshape (self.target_.values ) # for consistency 
            
        return self 

    def plotClusterParallelCoords (
            self, 
            classes: List [Any] = None, 
            pkg = 'pd',
            rxlabel: int =45 , 
            **kwd
            )->'QuestPlotter': 
        """ Use parallel coordinates in multivariates for clustering 
        visualization  
        
        Parameters 
        ------------
        classes: list, default: None
            a list of class names for the legend The class labels for each 
            class in y, ordered by sorted class index. These names act as a 
            label encoder for the legend, identifying integer classes or 
            renaming string labels. If omitted, the class labels will be taken 
            from the unique values in y.
            
            Note that the length of this list must match the number of unique 
            values in y, otherwise an exception is raised.
        pkg: str, Optional, 
            kind or library to use for visualization. can be ['sns'|'pd'] for 
            'yellowbrick' or 'pandas' respectively. *default* is ``pd``.
            
        rxlabel: int, default is ``45`` 
            rotate the xlabel when using pkg is set to ``pd``. 
            
        kws: dict, 
            Additional keywords arguments are passed down to 
            :class:`yellowbrick.ParallelCoordinates` and
            :func:`pandas.plotting.parallel_coordinates`
            
        Returns 
        --------
        ``self``: `QuestPlotter` instance and returns ``self`` for easy method chaining. 
        
        Examples
        --------
        >>> from gofast.datasets import fetch_data 
        >>> from gofast.view import QuestPlotter 
        >>> data =fetch_data('original data').get('data=dfy1')
        >>> p = QuestPlotter (target_name ='flow').fit(data)
        >>> p.plotClusterParallelCoords(pkg='yb')
        ... <'QuestPlotter':xname=None, yname=None , target_name='flow'>
         
        """
        self.inspect 
        
        if str(pkg) in ('yellowbrick', 'yb'): 
            pkg ='yb'
        else: pkg ='pd' 
        
        fig, ax = plt.subplots(figsize = self.fig_size )
        
        df = self.data .copy() 
        df = select_features(df, include ='number')
        
        if pkg =='yb': 
            import_optional_dependency('yellowbrick', (
                "Cannot plot 'parallelcoordinates' with missing"
                " 'yellowbrick'package.")
                )
            pc =ParallelCoordinates(ax =ax , 
                                    features = df.columns, 
                                    classes =classes , 
                                    **kwd 
                                    )
            pc.fit(df, y = None or self.y_)
            pc.transform (df)
            label_format = '{:.0f}'
            
            ticks_loc = list(ax.get_xticks())
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax.set_xticklabels([label_format.format(x) for x in ticks_loc], 
                                rotation =rxlabel)
            pc.show()
            
        elif pkg =='pd': 
            if self.target_name is not None: 
                if self.target_name not in df.columns :
                    df[self.target_name ]= self.y_ 
            parallel_coordinates(df, class_column= self.target_name, 
                                 ax= ax, **kwd
                                 )
        self.save (fig)
        
        return self 
    
    def plotRadViz (self,
                    classes: List [Any] = None, 
                    pkg:str = 'pd',
                    **kwd
                    )-> 'QuestPlotter': 
        """ plot each sample on circle or square, with features on the  
        circonference to vizualize separately between target. 
        
        Values are normalized and each figure has a spring that pulls samples 
        to it based on the value. 
        
        Parameters 
        ------------
        
        classes: list of int | float, [categorized classes] 
            must be a value in the target.  Specified classes must match 
            the number of unique values in target. otherwise an error occurs.
            the default behaviour  i.e. ``None`` detect all classes in unique  
            value in the target. 
            
        pkg: str, Optional, 
            kind or library to use for visualization. can be ['sns'|'pd']  for 
             'yellowbrick' or 'pandas' respectively. default is ``pd``.   
        kws: dict, 
            Additional keywords arguments are passed down to 
            :class:`yellowbrick.RadViZ` and :func:`pandas.plotting.radviz`
            
        Returns 
        -----------
        ``self``: `QuestPlotter` instance and returns ``self`` for easy method chaining. 
        
        Examples 
        ---------
        (1)-> using yellowbrick RadViz 
        
        >>> from gofast.datasets import fetch_data 
        >>> from gofast.view import QuestPlotter 
        >>> data0 = fetch_data('bagoue original').get('data=dfy1')
        >>> p = QuestPlotter(target_name ='flow').fit(data0)
        >>> p.plotRadViz(classes= [0, 1, 2, 3] ) # can set to None 
        
        (2) -> Using pandas radviz plot 
        
        >>> # use pandas with 
        >>> data2 = fetch_data('bagoue original').get('data=dfy2')
        >>> p = QuestPlotter(target_name ='flow').fit(data2)
        >>> p.plotRadViz(classes= None, pkg='pd' )
        ... <'QuestPlotter':xname=None, yname=None , target_name='flow'>
        """
        self.inspect 
        
        fig, ax = plt.subplots(figsize = self.fig_size )
        df = self.data .copy() 
        
        if str(pkg) in ('yellowbrick', 'yb'): 
            pkg ='yb'
        else: pkg ='pd' 
        
        if classes is None :  
            if self.target_name is None: 
                raise TypeError (
                    "target name is missing. Can not fetch the target."
                    " Provide the target name instead."
                    )
            classes = list(np.unique (self.y_))
            
        df = select_features(df, include ='number')

        if pkg =='yb': 
            rv = RadViz( ax = ax , 
                        classes = classes , 
                        features = df.columns,
                        **kwd 
                        )
            rv.fit(df, y =None or self.y_ )
            _ = rv.transform(df )
            
            rv.show()
            
        elif pkg =='pd': 
            if (self.target_name is not None)  and (self.y_ is not None) :
                df [self.target_name] = self.y_ 
            radviz (df , class_column= self.target_name , ax = ax, 
                    **kwd 
                    )
            
        self.save (fig)
        
        return self 

    def plotPairwiseFeatures(
        self ,  
        corr:str = 'pearson', 
        pkg:str ='sns', 
        **kws
        )-> 'QuestPlotter': 
        """ Create pairwise comparizons between features. 
        
        Plots shows a ['pearson'|'spearman'|'covariance'] correlation. 
        
        Parameters 
        -----------
        corr: str, ['pearson'|'spearman'|'covariance']
            Method of correlation to perform. Note that the 'person' and 
            'covariance' don't support string value. If such kind of data 
            is given, turn the `corr` to `spearman`. 
            *default* is ``pearson``
        
        pkg: str, Optional, 
            kind or library to use for visualization. can be ['sns'|'yb']  for 
            'seaborn' or 'yellowbrick' respectively. default is ``sns``.   
        kws: dict, 
            Additional keywords arguments are passed down to 
            :class:`yellowbrick.Rand2D` and `seaborn.heatmap`
        
        Returns 
        -----------
        ``self``: `QuestPlotter` instance and returns ``self`` for easy method chaining.
             
        Example 
        ---------
        >>> from gofast.datasets import fetch_data 
        >>> from gofast.view import QuestPlotter 
        >>> data = fetch_data ('bagoue original').get('data=dfy1') 
        >>> p= QuestPlotter(target_name='flow').fit(data)
        >>> p.plotPairwiseFeatures(fmt='.2f', corr='spearman', pkg ='yb',
                                     annot=True, 
                                     cmap='RdBu_r', 
                                     vmin=-1, 
                                     vmax=1 )
        ... <'QuestPlotter':xname='sfi', yname='ohmS' , target_name='flow'>
        """
        self.inspect 
        
        if str(pkg) in ('yellowbrick', 'yb'): 
            pkg ='yb'
        else: pkg ='sns' 
  
        fig, ax = plt.subplots(figsize = self.fig_size )
        df = self.data .copy() 
        
        if pkg =='yb': 
            pcv = Rank2D( ax = ax, 
                         features = df.columns, 
                         algorithm=corr, **kws)
            pcv.fit(df, y = None or self.y_ )
            pcv.transform(df)
            pcv.show() 
            
        elif pkg =='sns': 
            sns.set(rc={"figure.figsize":self.fig_size}) 
            fig = sns.heatmap(data =df.corr() , **kws
                             )
        
        self.save (fig)
        
        return self 
        
    def plotCutQuantiles(
            self, 
            xname: str =None, 
            yname:str =None, 
            q:int =10 , 
            bins: int=3 , 
            cmap:str = 'viridis',
            duplicates:str ='drop', 
            **kws
        )->'QuestPlotter': 
        """Compare the cut or `q` quantiles values of ordinal categories.
        
        It simulates that the the bining of 'xname' into a `q` quantiles, and 
        'yname'into `bins`. Plot is normalized so its fills all the vertical area.
        which makes easy to see that in the `4*q %` quantiles.  
        
        Parameters 
        -------------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
        q: int or list-like of float
            Number of quantiles. 10 for deciles, 4 for quartiles, etc. 
            Alternately array of quantiles, e.g. [0, .25, .5, .75, 1.] for 
            quartiles.
        bins: int, sequence of scalars, or IntervalIndex
            The criteria to bin by.

            * int : Defines the number of equal-width bins in the range of x. 
                The range of x is extended by .1% on each side to include the 
                minimum and maximum values of x.

            * sequence of scalars : Defines the bin edges allowing for non-uniform 
                width. No extension of the range of x is done.

            * IntervalIndex : Defines the exact bins to be used. Note that 
                IntervalIndex for bins must be non-overlapping.
                
        labels: array or False, default None
            Used as labels for the resulting bins. Must be of the same length 
            as the resulting bins. If False, return only integer indicators of 
            the bins. If True, raises an error.
            
        cmap: str, color or list of color, optional
            The matplotlib colormap  of the bar faces.
            
        duplicates: {default 'raise', 'drop}, optional
            If bin edges are not unique, raise ValueError or drop non-uniques.
            *default* is 'drop'
        kws: dict, 
            Other keyword arguments are passed down to `pandas.qcut` .
            
        Returns 
        -------
        ``self``: `QuestPlotter` instance and returns ``self`` for easy method chaining.
        
        Examples 
        ---------
        >>> from gofast.datasets import fetch_data 
        >>> from gofast.view import QuestPlotter 
        >>> data = fetch_data ('bagoue original').get('data=dfy1') 
        >>> p= QuestPlotter(target_name='flow').fit(data)
        >>> p.plotCutQuantiles(xname ='sfi', yname='ohmS')
        """
        self.inspect 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 
        
        fig, ax = plt.subplots(figsize = self.fig_size )
        
        df = self.data .copy() 
        (df.assign(
            xname_bin = pd.qcut( 
                df[self.xname_], q = q, labels =False, 
                duplicates = duplicates, **kws
                ),
            yname_bin = pd.cut(
                df[self.yname_], bins =bins, labels =False, 
                duplicates = duplicates,
                ), 
            )
        .groupby (['xname_bin', 'yname_bin'])
        .size ().unstack()
        .pipe(lambda df: df.div(df.sum(1), axis=0))
        .plot.bar(stacked=True, 
                  width=1, 
                  ax=ax, 
                  cmap =cmap)
        .legend(bbox_to_anchor=(1, 1))
        )
                  
        self.save(fig)
        return self 
         
    def plotDistributions(
            self, 
            xname: str =None, 
            yname:str =None, 
            kind:str ='box',
            **kwd
        )->'QuestPlotter': 
        """Visualize distributions using the box, boxen or violin plots. 
        
        Parameters 
        -----------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
            
        kind: str 
            style of the plot. Can be ['box'|'boxen'|'violin']. 
            *default* is ``box``
            
        kwd: dict, 
            Other keyword arguments are passed down to `seaborn.boxplot` .
            
        Returns 
        -----------
        ``self``: `QuestPlotter` instance and returns ``self`` for easy 
        method chaining.
        
        Example
        --------
        >>> from gofast.datasets import fetch_data 
        >>> from gofast.view import QuestPlotter 
        >>> data = fetch_data ('bagoue original').get('data=dfy1') 
        >>> p= QuestPlotter(target_name='flow').fit(data)
        >>> p.plotDistributions(xname='flow', yname='sfi', kind='violin')
        
        """
    
        self.inspect 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 
        
        kind = str(kind).lower() 
    
        if kind.find('violin')>=0: kind = 'violin'
        elif kind.find('boxen')>=0 : kind ='boxen'
        else : kind ='box'
        
        df = self.data.copy() 
        if (self.target_name not in df.columns) and (self.y_ is not None): 
            df [self.target_name] = self.y_  
        
        if kind =='box': 
            g= sns.boxplot( 
                data = df , 
                x= self.xname_,
                y=self.yname_ , 
                **kwd
                )
        if kind =='boxen': 
            g= sns.boxenplot( 
                data = df , 
                x= self.xname_, 
                y=self.yname_ , 
                **kwd
                )
        if kind =='violin': 
            g = sns.violinplot( 
                data = df , 
                x= self.xname_, 
                y=self.yname_ , 
                **kwd
                )
        
        self.save(g)
        
        return self 
    
    
    def plotPairGrid (
        self, 
        xname: str =None, 
        yname:str =None, 
        vars: List[str]= None, 
        **kwd 
    ) -> 'QuestPlotter': 
        """ Create a pair grid. 
        
        Is a matrix of columns and kernel density estimations. To color by a 
        columns from a dataframe, use 'hue' parameter. 
        
        Parameters 
        -------------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
        vars: list, str 
            list of items in the dataframe columns. Raise an error if items 
            dont exist in the dataframe columns. 
        kws: dict, 
            Other keyword arguments are passed down to `seaborn.joinplot`_ .
            
        Returns 
        -----------
        ``self``: `QuestPlotter` instance and returns ``self`` for easy method chaining.
            
        Example
        --------
        >>> from gofast.datasets import fetch_data 
        >>> from gofast.plot import QuestPlotter 
        >>> data = fetch_data ('bagoue original').get('data=dfy1') 
        >>> p= QuestPlotter(target_name='flow').fit(data)
        >>> p.plotPairGrid (vars = ['magnitude', 'power', 'ohmS'] ) 
        ... <'QuestPlotter':xname=(None,), yname=None , target_name='flow'>
        
        """
        self.inspect 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 

        df = self.data.copy() 
        if (self.target_name not in df.columns) and (self.y_ is not None): 
            df [self.target_name] = self.y_ # set new dataframe with a target
        if vars is None : 
            vars = [self.xname_, self.y_name ]
            
        sns.set(rc={"figure.figsize":self.fig_size}) 
        g = sns.pairplot (df, vars= vars, hue = self.target_name, 
                            **kwd, 
                             )
        self.save(g)
        
        return self 
    
    def plotFancierJoin(
            self, 
            xname: str, 
            yname:str =None,  
            corr: str = 'pearson', 
            kind:str ='scatter', 
            pkg='sns', 
            yb_kws =None, 
            **kws
    )->'QuestPlotter': 
        """ fancier scatterplot that includes histogram on the edge as well as 
        a regression line called a `joinplot` 
        
        Parameters 
        -------------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
        pkg: str, Optional, 
            kind or library to use for visualization. can be ['sns'|'yb']  for 
            'seaborn' or 'yellowbrick'. default is ``sns``.
            
        kind : str in {'scatter', 'hex'}, default: 'scatter'
            The type of plot to render in the joint axes. Note that when 
            kind='hex' the target cannot be plotted by color.
            
        corr: str, default: 'pearson'
            The algorithm used to compute the relationship between the 
            variables in the joint plot, one of: 'pearson', 'covariance', 
            'spearman', 'kendalltau'.
            
        yb_kws: dict, 
            Additional keywords arguments from 
            :class:`yellowbrick.JointPlotVisualizer`
        kws: dict, 
            Other keyword arguments are passed down to `seaborn.joinplot`_ .
            
        Returns 
        -----------
        ``self``: `QuestPlotter` instance and returns ``self`` for easy method chaining.
             
        Notes 
        -------
        When using the `yellowbrick` library and array i.e a (x, y) variables 
        in the columns as well as the target arrays must not contain infs or NaNs
        values. A value error raises if that is the case. 
        
        .. _seaborn.joinplot: https://seaborn.pydata.org/generated/seaborn.joinplot.html
        
        """
        pkg = str(pkg).lower().strip() 
        if pkg in ('yb', 'yellowbrick'): pkg ='yb'
        else:  pkg ='sns'
        
        self.inspect 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 
        
        # assert yb_kws arguments 
        yb_kws = yb_kws or dict() 
        yb_kws = _assert_all_types(yb_kws, dict)
        
        if pkg =='yb': 
            fig, ax = plt.subplots(figsize = self.fig_size )
            jpv = JointPlotVisualizer(
                ax =ax , 
                #columns =self.xname_,   # self.data.columns, 
                correlation=corr, 
                # feature=self.xname_, 
                # target=self.target_name, 
                kind= kind , 
                fig = fig, 
                **yb_kws
                )
            jpv.fit(
                self.data [self.xname_], 
                self.data [self.target_name] if self.y_ is None else self.y_,
                    ) 
            jpv.show()
        elif pkg =='sns': 
            sns.set(rc={"figure.figsize":self.fig_size}) 
            sns.set_style (self.sns_style)
            df = self.data.copy() 
            if (self.target_name not in df.columns) and (self.y_ is not None): 
                df [self.target_name] = self.y_ # set new dataframe with a target 
                
            fig = sns.jointplot(
                data= df, 
                x = self.xname_, 
                y= self.yname_,
                **kws
                ) 
            
        self.save(fig )
        
        return self 
        
    def plotScatter (
            self, 
            xname:str =None, 
            yname:str = None, 
            c:str |Series|ArrayLike =None, 
            s: int |ArrayLike =None, 
            **kwd
        )->'QuestPlotter': 
        """ Shows the relationship between two numeric columns. 
        
        Parameters 
        ------------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
          
        c: str, int or array_like, Optional
            The color of each point. Possible values are:
                * A single color string referred to by name, RGB or RGBA code,
                     for instance 'red' or '#a98d19'.
                * A sequence of color strings referred to by name, RGB or RGBA 
                    code, which will be used for each point’s color recursively.
                    For instance [‘green’,’yellow’] all points will be filled 
                    in green or yellow, alternatively.
                * A column name or position whose values will be used to color 
                    the marker points according to a colormap.
                    
        s: scalar or array_like, Optional, 
            The size of each point. Possible values are:
                * A single scalar so all points have the same size.
                * A sequence of scalars, which will be used for each point’s 
                    size recursively. For instance, when passing [2,14] all 
                    points size will be either 2 or 14, alternatively.
        kwd: dict, 
            Other keyword arguments are passed down to `seaborn.scatterplot`_ .
            
        Returns 
        -----------
        ``self``: `QuestPlotter` instance 
            returns ``self`` for easy method chaining.

        Example 
        ---------
        >>> from gofast.view import QuestPlotter 
        >>> p = QuestPlotter(target_name='flow').fit(data).plotScatter (
            xname ='sfi', yname='ohmS')
        >>> p
        ...  <'QuestPlotter':xname='sfi', yname='ohmS' , target_name='flow'>
        
        References 
        ------------
        Scatterplot: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
        Pd.scatter plot: https://www.w3resource.com/pandas/dataframe/dataframe-plot-scatter.php
        
        .. _seaborn.scatterplot: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
        
        """
        self.inspect 
        
        hue = kwd.pop('hue', None) 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 
        
        if hue is not None: 
            self.target_name = hue 

        if xname is not None: 
            exist_features( self.data, self.xname_ )
        if yname is not None: 
            exist_features( self.data, self.yname_ )
        
        # state the fig plot and change the figure size 
        sns.set(rc={"figure.figsize":self.fig_size}) #width=3, #height=4
        if self.sns_style is not None: 
           sns.set_style(self.sns_style)
        # try : 
        fig= sns.scatterplot( data = self.data, x = self.xname_,
                             y=self.yname_, hue =self.target_name, 
                        # ax =ax , # call matplotlib.pyplot.gca() internally
                        **kwd)
        # except : 
        #     warnings.warn("The following variable cannot be assigned with "
        #                   "wide-form data: `hue`; use the pandas scatterplot "
        #                   "instead.")
        #     self.data.plot.scatter (x =xname , y=yname, c=c,
        #                             s = s,  ax =ax  )
        
        self.save(fig) 
        
        return self 
    
    def plotHistBinTarget (
            self, 
            xname: str, 
            c: Any =None, *,  
            posilabel: str = None, 
            neglabel: str= None,
            kind='binarize', 
            **kws
        )->'QuestPlotter': 
        """
        A histogram of continuous against the target of binary plot. 
        
        Parameters 
        ----------
        xname: str, 
            the column name to consider on x-axis. Shoud be  an item in the  
            dataframe columns. Raise an error if element does not exist.  
            
        c: str or  int  
            the class value in `y` to consider. Raise an error if not in `y`. 
            value `c` can be considered as the binary positive class 
            
        posilabel: str, Optional 
            the label of `c` considered as the positive class 
        neglabel: str, Optional
            the label of other classes (categories) except `c` considered as 
            the negative class 
        kind: str, Optional, (default, 'binarize') 
            the kind of plot features against target. `binarize` considers 
            plotting the positive class ('c') vs negative class ('not c')
        
        kws: dict, 
            Additional keyword arguments of  `seaborn displot`_ 
            
        Returns 
        -----------
        ``self``: `QuestPlotter` instance 
            returns ``self`` for easy method chaining.

        Examples
        --------
        >>> from gofast.utils import read_data 
        >>> from gofast.view import QuestPlotter
        >>> data = read_data  ( 'data/geodata/main.bagciv.data.csv' ) 
        >>> p = QuestPlotter(target_name ='flow').fit(data)
        >>> p.fig_size = (7, 5)
        >>> p.savefig ='bbox.png'
        >>> p.plotHistBinTarget (xname= 'sfi', c = 0, kind = 'binarize',  kde=True, 
                          posilabel='dried borehole (m3/h)',
                          neglabel = 'accept. boreholes'
                          )
        Out[95]: <'QuestPlotter':xname='sfi', yname=None , target_name='flow'>
        """
     
        self.inspect 
            
        self.xname_ = xname or self.xname_ 
        
        exist_features(self.data, self.xname_) # assert the name in the columns
        df = self.data.copy() 
       
        if str(kind).lower().strip().find('bin')>=0: 
            if c  is None: 
                raise ValueError ("Need a categorical class value for binarizing")
                
            _assert_all_types(c, float, int)
            
            if self.y_ is None: 
                raise ValueError ("target name is missing. Specify the `target_name`"
                                  f" and refit {self.__class__.__name__!r} ")
            if not _isin(self.y_, c ): 
                raise ValueError (f"c-value should be a class label, got '{c}'"
                                  )
            mask = self.y_ == c 
            # for consisteny use np.unique to get the classes
            
            neglabel = neglabel or shrunkformat( 
                np.unique (self.y_[~(self.y_ == c)]) 
                )

        else: 
  
            if self.target_name is None: 
                raise ValueError("Can't plot binary classes with missing"
                                 " target name ")
            df[self.target_name] = df [self.target_name].map(
                lambda x : 1 if ( x == c if isinstance(c, str) else x <=c 
                                 )  else 0  # mapping binary target
                )

        #-->  now plot 
        # state the fig plot  
        sns.set(rc={"figure.figsize":self.fig_size}) 
        if  str(kind).lower().strip().find('bin')>=0: 
            g=sns.histplot (data = df[mask][self.xname_], 
                            label= posilabel or str(c) , 
                            linewidth = self.lw, 
                            **kws
                          )
            
            g=sns.histplot( data = df[~mask][self.xname_], 
                              label= neglabel , 
                              linewidth = self.lw,
                              **kws,
                              )
        else : 
            g=sns.histplot (data =df , 
                              x = self.xname_,
                              hue= self.target_name,
                              linewidth = self.lw, 
                          **kws
                        )
            
        if self.sns_style is not None: 
            sns.set_style(self.sns_style)
            
        g.legend ()
        
        # self.save(g)
  
        return self 

    def plotHist(self,xname: str = None, *,  kind:str = 'hist', 
                   **kws 
                   ): 
        """ A histogram visualization of numerica data.  
        
        Parameters 
        ----------
        xname: str , xlabel 
            feature name in the dataframe  and is the label on x-axis. 
            Raises an error , if it does not exist in the dataframe 
        kind: str 
            Mode of pandas series plotting. the *default* is ``hist``. 
            
        kws: dict, 
            additional keywords arguments from : func:`pandas.DataFrame.plot` 
            
        Return
        -------
        ``self``: `QuestPlotter` instance 
            returns ``self`` for easy method chaining.
            
        """
        self.inspect  
        self.xname_ = xname or self.xname_ 
        xname = _assert_all_types(self.xname_,str )
        # assert whether whether  feature exists 
        exist_features(self.data, self.xname_)
    
        fig, ax = plt.subplots (figsize = self.fig_size or self.fig_size )
        self.data [self.xname_].plot(kind = kind , ax= ax  , **kws )
        
        self.save(fig)
        
        return self 
    
    def plotMissingPatterns(self, *, 
                kind: str =None, 
                sample: float = None,  
                **kwd
                ): 
        """
        Vizualize patterns in the missing data.
        
        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
    
        kind: str, Optional 
            kind of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar`` plot 
            for dendrogram , :mod:`msno` bar and :mod:`plt` visualization 
            respectively: 
                
            * ``bar`` plot counts the  nonmissing data  using pandas
            * ``mbar`` use the :mod:`msno` package to count the number 
                of nonmissing data. 
            * dendrogram`` show the clusterings of where the data is missing. 
                leaves that are the same level predict one onother presence 
                (empty of filled). The vertical arms are used to indicate how  
                different cluster are. short arms mean that branch are 
                similar. 
            * ``corr` creates a heat map showing if there are correlations 
                where the data is missing. In this case, it does look like 
                the locations where missing data are corollated.
            * ``mpatterns`` is the default vizualisation. It is useful for viewing 
                contiguous area of the missing data which would indicate that 
                the missing data is  not random. The :code:`matrix` function 
                includes a sparkline along the right side. Patterns here would 
                also indicate non-random missing data. It is recommended to limit 
                the number of sample to be able to see the patterns. 
   
            Any other value will raise an error 
            
        sample: int, Optional
            Number of row to visualize. This is usefull when data is composed of 
            many rows. Skrunked the data to keep some sample for visualization is 
            recommended.  ``None`` plot all the samples ( or examples) in the data 
            
        kws: dict 
            Additional keywords arguments of :mod:`msno.matrix` plot. 
    
        Return
        -------
        ``self``: `QuestPlotter` instance 
            returns ``self`` for easy method chaining.
            
        Example
        --------
        >>> import pandas as pd 
        >>> from gofast.view import QuestPlotter
        >>> data = pd.read_csv ('data/geodata/main.bagciv.data.csv' ) 
        >>> p = QuestPlotter().fit(data)
        >>> p.fig_size = (12, 4)
        >>> p.plotMissingPatterns(kind ='corr')
        
        """
        self.inspect 
            
        kstr =('dendrogram', 'bar', 'mbar', 'correlation', 'mpatterns')
        kind = str(kind).lower().strip() 
        
        regex = re.compile (r'none|dendro|corr|base|default|mbar|bar|mpat', 
                            flags= re.IGNORECASE)
        kind = regex.search(kind) 
        
        if kind is None: 
            raise ValueError (f"Expect {smart_format(kstr, 'or')} not: {kind!r}")
            
        kind = kind.group()
  
        if kind in ('none', 'default', 'base', 'mpat'): 
            kind ='mpat'
        
        if sample is not None: 
            sample = _assert_all_types(sample, int, float)
            
        if kind =='bar': 
            fig, ax = plt.subplots (figsize = self.fig_size, **kwd )
            (1- self.data.isnull().mean()).abs().plot.bar(ax=ax)
    
        elif kind  in ('mbar', 'dendro', 'corr', 'mpat'): 
            try : 
                msno 
            except : 
                raise ModuleNotFoundError(
                    f"Missing 'missingno' package. Can not plot {kind!r}")
                
            if kind =='mbar': 
                ax = msno.bar(
                    self.data if sample is None else self.data.sample(sample),
                              figsize = self.fig_size 
                              )
    
            elif kind =='dendro': 
                ax = msno.dendrogram(self.data, figsize = self.fig_size , **kwd) 
        
            elif kind =='corr': 
                ax= msno.heatmap(self.data, figsize = self.fig_size)
            else : 
                ax = msno.matrix(
                    self.data if sample is None else self.data.sample (sample),
                                 figsize= self.fig_size , **kwd)
        
        if self.savefig is not None:
            fig.savefig(self.savefig, dpi =self.fig_dpi 
                        ) if kind =='bar' else ax.get_figure (
                ).savefig (self.savefig,  dpi =self.fig_dpi)
        
        return self 
    
    def __repr__(self): 
        """ Represent the output class format """
        return  "<{0!r}:xname={1!r}, yname={2!r} , target_name={3!r}>".format(
            self.__class__.__name__, self.xname_ , self.yname_ , self.target_name 
            )
              
class EasyPlotter (BasePlot): 
    """
    Special class dealing with analysis modules for quick diagrams, 
    histograms and bar visualizations. 
    
    Originally, it was designed for the flow rate prediction, however, it still 
    works with any other dataset by following the parameters details. 
      
    Parameters 
    -------------
    {params.core.data}
    {params.core.y}
    {params.core.target_name}
    {params.qdoc.classes}
    {params.qdoc.mapflow}
    {params.base.savefig}
    {params.base.fig_dpi}
    {params.base.fig_num}
    {params.base.fig_size}
    {params.base.fig_orientation}
    {params.base.fig_title}
    {params.base.fs}
    {params.base.ls}
    {params.base.lc}
    {params.base.lw}
    {params.base.alpha}
    {params.base.font_weight}
    {params.base.font_style}
    {params.base.font_size}
    {params.base.ms}
    {params.base.marker}
    {params.base.marker_facecolor}
    {params.base.marker_edgecolor}
    {params.base.marker_edgewidth}
    {params.base.xminorticks}
    {params.base.yminorticks}
    {params.base.bins}
    {params.base.xlim}
    {params.base.ylim}
    {params.base.xlabel}
    {params.base.ylabel}
    {params.base.rotate_xlabel}
    {params.base.rotate_ylabel}
    {params.base.leg_kws}
    {params.base.plt_kws}
    {params.base.glc}
    {params.base.glw}
    {params.base.galpha}
    {params.base.gaxis}
    {params.base.gwhich}
    {params.base.tp_axis}
    {params.base.tp_labelsize}
    {params.base.tp_bottom}
    {params.base.tp_labelbottom}
    {params.base.tp_labeltop}
    {params.base.cb_orientation}
    {params.base.cb_aspect}
    {params.base.cb_shrink}
    {params.base.cb_pad}
    {params.base.cb_anchor}
    {params.base.cb_panchor}
    {params.base.cb_label}
    {params.base.cb_spacing}
    {params.base.cb_drawedges} 
    {params.sns.sns_orient}
    {params.sns.sns_style}
    {params.sns.sns_palette}
    {params.sns.sns_height}
    {params.sns.sns_aspect}
    
    Returns
    --------
    {returns.self}
    
    Examples
    ---------
    >>> from gofast.plot.explore import  EasyPlotter 
    >>> data = 'data/geodata/main.bagciv.data.csv'
    >>> qkObj = EasyPlotter(  leg_kws= dict( loc='upper right'),
    ...          fig_title = '`sfi` vs`ohmS|`geol`',
    ...            ) 
    >>> qkObj.target_name='flow' # target the DC-flow rate prediction dataset
    >>> qkObj.mapflow=True  # to hold category FR0, FR1 etc..
    >>> qkObj.fit(data) 
    >>> sns_pkws= dict ( aspect = 2 , 
    ...          height= 2, 
    ...                  )
    >>> map_kws= dict( edgecolor="w")    
    >>> qkObj.plotDiscussingFeatures(features =['ohmS', 'sfi','geol', 'flow'],
    ...                           map_kws=map_kws,  **sns_pkws
    ...                         )   
    """.format(
        params=_param_docs,
        returns= _core_docs["returns"],
    )
   
    def __init__(
        self,  
        classes = None, 
        target_name= None, 
        mapflow=False, 
        **kws
        ): 
        super().__init__(**kws)
        
        self._logging =gofastlog().get_gofast_logger(self.__class__.__name__)
        self.classes=classes
        self.target_name=target_name
        self.mapflow=mapflow
        
    @property 
    def data(self): 
        return self.data_ 
    
    @data.setter 
    def data (self, data):
        """ Read the data file and separate the target from the features 
        if  the target name `target_names` is given.
        """
        self.data_ = is_readable(
            data , input_name="'data'")
        
        if str(self.target_name).lower() in self.data_.columns.str.lower(): 
            ix = list(self.data.columns.str.lower()).index (
                self.target_name.lower() )
            self.y = self.data_.iloc [:, ix ]

            self.X_ = self.data_.drop(columns =self.data_.columns[ix] , 
                                         )
            
    def fit(
        self,
        data: str | DataFrame, 
        y: Optional[Series| ArrayLike]=None
    )-> "EasyPlotter" : 
        """ 
        Fit data and populate the attributes for plotting purposes. 
        
        Parameters 
        ----------
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target.  
        
        y: array-like, optional 
            array of the target. Must be the same length as the data. If `y` 
            is provided and `data` is given as ``str`` or ``DataFrame``, 
            all the data should be considered as the X data for analysis. 
  
         Returns
         -------
         self: :class:`EasyPlotter` instance
             Returns ``self`` for easy method chaining.
             
        Examples 
        --------
        >>> from gofast.datasets import load_bagoue
        >>> data = load_bagoue ().frame
        >>> from gofast.plot.explore  import EasyPlotter
        >>> qplotObj= EasyPlotter(xlabel = 'Flow classes in m3/h',
                                ylabel='Number of  occurence (%)')
        >>> qplotObj.target_name= None # eith nameof target set to None 
        >>> qplotObj.fit(data)
        >>> qplotObj.data.iloc[1:2, :]
        ...     num name      east      north  ...         ohmS        lwi      geol flow
            1  2.0   b2  791227.0  1159566.0  ...  1135.551531  21.406531  GRANITES  0.0
        >>> qplotObj.target_name= 'flow'
        >>> qplotObj.mapflow= True # map the flow from num. values to categ. values
        >>> qplotObj.fit(data)
        >>> qplotObj.data.iloc[1:2, :]
        ...    num name      east      north  ...         ohmS        lwi      geol flow
            1  2.0   b2  791227.0  1159566.0  ...  1135.551531  21.406531  GRANITES  FR0
         
        """
        self.data = data 
        if y is not None: 
            _, y = check_X_y(
                self.data, y, 
                force_all_finite="allow-nan", 
                dtype =object, 
                to_frame = True 
                )
            y = _assert_all_types(y, np.ndarray, list, tuple, pd.Series)
            if len(y)!= len(self.data) :
                raise ValueError(
                    f"y and data must have the same length but {len(y)} and"
                    f" {len(self.data)} were given respectively.")
            
            self.y = pd.Series (y , name = self.target_name or 'none')
            # for consistency get the name of target 
            self.target_name = self.y.name 
            
        return self 
    
    def plotHistCatDistribution(
        self, 
        stacked: bool = False,  
        **kws
        ): 
        """
        Histogram plot of category distributions.
        
        Plots a distribution of categorized classes according to the 
        percentage of occurence. 
        
        Parameters 
        -----------
        stacked: bool 
            Pill bins one to another as a cummulative values. *default* is 
            ``False``. 
            
        bins : int, optional 
             contains the integer or sequence or string
             
        range : list, optional 
            is the lower and upper range of the bins
        
        density : bool, optional
             contains the boolean values 
            
        weights : array-like, optional
            is an array of weights, of the same shape as `data`
            
        bottom : float, optional 
            is the location of the bottom baseline of each bin
            
        histtype : str, optional 
            is used to draw type of histogram. {'bar', 'barstacked', step, 'stepfilled'}
            
        align : str, optional
             controls how the histogram is plotted. {'left', 'mid', 'right'}
             
        rwidth : float, optional,
            is a relative width of the bars as a fraction of the bin width
            
        log : bool, optional
            is used to set histogram axis to a log scale
            
        color : str, optional 
            is a color spec or sequence of color specs, one per dataset
            
        label : str , optional
            is a string, or sequence of strings to match multiple datasets
            
        normed : bool, optional
            an optional parameter and it contains the boolean values. It uses 
            the density keyword argument instead.
          
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`EasyPlotter` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `EasyPlotter` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
        
        Examples 
        ---------
        >>> from gofast.plot.explore  import EasyPlotter
        >>> from gofast.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qplotObj= EasyPlotter(xlabel = 'Flow classes',
                                ylabel='Number of  occurence (%)',
                                lc='b', target_name='flow')
        >>> qplotObj.sns_style = 'darkgrid'
        >>> qplotObj.fit(data)
        >>> qplotObj. plotHistCatDistribution()
        
        """
        self._logging.info('Quick plot of categorized classes distributions.'
                           f' the target name: {self.target_name!r}')
        
        self.inspect 
    
        if self.target_name is None and self.y is None: 
            raise FeatureError(
                "Please specify 'target_name' as the name of the target")

        # reset index 
        df_= self.data_.copy()  #make a copy for safety 
        df_.reset_index(inplace =True)
        
        if kws.get('bins', None) is not None: 
            self.bins = kws.pop ('bins', None)
            
        plt.figure(figsize =self.fig_size)
        plt.hist(df_[self.target_name], bins=self.bins ,
                  stacked = stacked , color= self.lc , **kws)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.fig_title)

        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation
                        )
        
        return self 
    
    def plotBarCatDistribution(
        self,
        basic_plot: bool = True,
        groupby: List[str] | Dict [str, float] =None,
        **kws):
        """
        Bar plot distribution. 
        
        Plots a categorical distribution according to the occurence of the 
        `target` in the data. 
        
        Parameters 
        -----------
        basic_pot: bool, 
            Plot only the occurence of targetted columns from 
            `matplotlib.pyplot.bar` function. 
            
        groupby: list or dict, optional 
            Group features for plotting. For instance it plot others features 
            located in the df columns. The plot features can be on ``list``
            and use default plot properties. To customize plot provide, one may 
            provide, the features on ``dict`` with convenients properties 
            like::

                * `groupby`= ['shape', 'type'] #{'type':{'color':'b',
                                             'width':0.25 , 'sep': 0.}
                                     'shape':{'color':'g', 'width':0.25, 
                                             'sep':0.25}}
        kws: dict, 
            Additional keywords arguments from `seaborn.countplot`
          
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`EasyPlotter` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `EasyPlotter` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
        
        Examples
        ----------
        >>> from gofast.plot.explore  import EasyPlotter
        >>> from gofast.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qplotObj= EasyPlotter(xlabel = 'Anomaly type',
                                ylabel='Number of  occurence (%)',
                                lc='b', target_name='flow')
        >>> qplotObj.sns_style = 'darkgrid'
        >>> qplotObj.fit(data)
        >>> qplotObj. plotBarCatDistribution(basic_plot =False, 
        ...                      groupby=['shape' ])
   
        """
        self.inspect 
        
        fig, ax = plt.subplots(figsize = self.fig_size)
        
        df_= self.data.copy(deep=True)  #make a copy for safety 
        df_.reset_index(inplace =True)
        
        if groupby is None:
            mess= ''.join([
                'Basic plot is turn to``False`` but no specific plot is', 
                "  detected. Please provide a specific column's into "
                " a `specific_plot` argument."])
            self._logging.debug(mess)
            warnings.warn(mess)
            basic_plot =True
            
        if basic_plot : 
            ax.bar(list(set(df_[self.target_name])), 
                        df_[self.target_name].value_counts(normalize =True),
                        label= self.fig_title, color = self.lc, )  
    
        if groupby is not None : 
            if hasattr(self, 'sns_style'): 
                sns.set_style(self.sns_style)
            if isinstance(groupby, str): 
                self.groupby =[groupby]
            if isinstance(groupby , dict):
                groupby =list(groupby.keys())
            for sll in groupby :
                ax= sns.countplot(x= sll,  hue=self.target_name, 
                                  data = df_, orient = self.sns_orient,
                                  ax=ax ,**kws)

        ax.set_xlabel(self. xlabel)
        ax.set_ylabel (self.ylabel)
        ax.set_title(self.fig_title)
        ax.legend() 
        
        if groupby is not None: 
            self._logging.info(
                'Multiple bar plot distribution grouped by  {0}.'.format(
                    generate_placeholders(groupby)).format(*groupby))
        
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        plt.show() if self.savefig is None else plt.close () 
        
        print('--> Bar distribution plot successfully done!'
              )if self.verbose > 0 else print()  
        
        return self 
    
    
    def plotMultiCatDistribution(
        self, 
        *, 
        x =None, 
        col=None, 
        hue =None, 
        targets: List[str]=None,
        x_features:List[str]=None ,
        y_features: List[str]=None, 
        kind:str='count',
        **kws): 
        """
        Figure-level interface for drawing multiple categorical distributions
        plots onto a FacetGrid.
        
        Multiple categorials plots  from targetted pd.series. 
        
        Parameters 
        -----------
        x, y, hue: list , Optional, 
            names of variables in data. Inputs for plotting long-form data. 
            See examples for interpretation. Here it can correspond to  
            `x_features` , `y_features` and `targets` from dataframe. Note that
            each columns item could be correspond as element of `x`, `y` or `hue`. 
            For instance x_features could refer to x-axis features and must be 
            more than 0 and set into a list. the `y_features` might match the 
            columns name for `sns.catplot`. If number of feature is more than 
            one, create a list to hold all features is recommended. 
            the `y` should fit the  `sns.catplot` argument ``hue``. Like other 
            it should be on list of features are greater than one. 
        
        row, colnames of variables in data, optional
            Categorical variables that will determine the faceting of the grid.
        
        col_wrapint
            "Wrap" the column variable at this width, so that the column facets 
            span multiple rows. Incompatible with a row facet.
        
        estimator: string or callable that maps vector -> scalar, optional
            Statistical function to estimate within each categorical bin.
        
        errorbar: string, (string, number) tuple, or callable
            Name of errorbar method (either "ci", "pi", "se", or "sd"), or a 
            tuple with a method name and a level parameter, or a function that
            maps from a vector to a (min, max) interval.
        
        n_bootint, optional
            Number of bootstrap samples used to compute confidence intervals.
        
        units: name of variable in data or vector data, optional
            Identifier of sampling units, which will be used to perform a 
            multilevel bootstrap and account for repeated measures design.
        
        seed: int, numpy.random.Generator, or numpy.random.RandomState, optional
            Seed or random number generator for reproducible bootstrapping.
        
        order, hue_order: lists of strings, optional
            Order to plot the categorical levels in; otherwise the levels are 
            inferred from the data objects.
        
        row_order, col_order: lists of strings, optional
            Order to organize the rows and/or columns of the grid in, otherwise
            the orders are inferred from the data objects.
        
        height: scalar
            Height (in inches) of each facet. See also: aspect.
        
        aspect:scalar
            Aspect ratio of each facet, so that aspect * height gives the width
            of each facet in inches.
        
        kind: str, optional
            `The kind of plot to draw, corresponds to the name of a categorical 
            axes-level plotting function. Options are: "strip", "swarm", "box", 
            "violin", "boxen", "point", "bar", or "count".
        
        native_scale: bool, optional
            When True, numeric or datetime values on the categorical axis 
            will maintain their original scaling rather than being converted 
            to fixed indices.
        
        formatter: callable, optional
            Function for converting categorical data into strings. Affects both
            grouping and tick labels.
        
        orient: "v" | "h", optional
            Orientation of the plot (vertical or horizontal). This is usually 
            inferred based on the type of the input variables, but it can be 
            used to resolve ambiguity when both x and y are numeric or when 
            plotting wide-form data.
        
        color: matplotlib color, optional
            Single color for the elements in the plot.
        
        palette: palette name, list, or dict
            Colors to use for the different levels of the hue variable. 
            Should be something that can be interpreted by color_palette(), 
            or a dictionary mapping hue levels to matplotlib colors.
        
        hue_norm: tuple or matplotlib.colors.Normalize object
            Normalization in data units for colormap applied to the hue 
            variable when it is numeric. Not relevant if hue is categorical.
        
        legend: str or bool, optional
            Set to False to disable the legend. With strip or swarm plots, 
            this also accepts a string, as described in the axes-level 
            docstrings.
        
        legend_out: bool
            If True, the figure size will be extended, and the legend will be 
            drawn outside the plot on the center right.
        
        share{x,y}: bool, 'col', or 'row' optional
            If true, the facets will share y axes across columns and/or x axes 
            across rows.
        
        margin_titles:bool
            If True, the titles for the row variable are drawn to the right of 
            the last column. This option is experimental and may not work in 
            all cases.
        
        facet_kws: dict, optional
            Dictionary of other keyword arguments to pass to FacetGrid.
        
        kwargs: key, value pairings
            Other keyword arguments are passed through to the underlying 
            plotting function.
            
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`EasyPlotter` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `EasyPlotter` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
        
        Examples
        ---------
        >>> from gofast.plot.explore  import EasyPlotter 
        >>> from gofast.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qplotObj= EasyPlotter(lc='b', target_name='flow')
        >>> qplotObj.sns_style = 'darkgrid'
        >>> qplotObj.mapflow=True # to categorize the flow rate 
        >>> qplotObj.fit(data)
        >>> fdict={
        ...            'x':['shape', 'type', 'type'], 
        ...            'col':['type', 'geol', 'shape'], 
        ...            'hue':['flow', 'flow', 'geol'],
        ...            } 
        >>> qplotObj.plotMultiCatDistribution(**fdict)
            
        """
        self.inspect 
            
        # set 
        if x is None :
            x = [None] 
        if col is None:
            col =[None] 
        if hue is None:
            hue =[None] 
        # for consistency put the values in list 
        x, col, hue = list(x) , list(col), list(hue)
        maxlen = max([len(i) for i in [x, col, hue]])  
        
        x.extend ( [None  for n  in range(maxlen - len(x))])
        col.extend ([None  for n  in range(maxlen - len(col))] )
        hue.extend ([None  for n  in range(maxlen - len(hue))])
       
        df_= self.data.copy(deep=True)
        df_.reset_index(inplace=True )
         
        if not hasattr(self, 'ylabel'): 
            self.ylabel= 'Number of  occurence (%)'
            
        if hue is not None: 
            self._logging.info(
                'Multiple categorical plots  from targetted {0}.'.format(
                    generate_placeholders(hue)).format(*hue))
        
        for ii in range(len(x)): 
            sns.catplot(data = df_,
                        kind= kind, 
                        x=  x[ii], 
                        col=col[ii], 
                        hue= hue[ii], 
                        linewidth = self.lw, 
                        height = self.sns_height,
                        aspect = self.sns_aspect,
                        **kws
                    ).set_ylabels(self.ylabel)
        
    
        plt.show()
       
        if self.sns_style is not None: 
            sns.set_style(self.sns_style)
            
        print('--> Multiple distribution plots sucessfully done!'
              ) if self.verbose > 0 else print()     
        
        return self 
    
    def plotCorrMatrix(
        self,
        cortype:str ='num',
        features: Optional[List[str]] = None, 
        method: str ='pearson',
        min_periods: int=1, 
        **sns_kws): 
        """
        Method to quick plot the numerical and categorical features. 
        
        Set `features` by providing the names of  features for visualization. 

        Parameters 
        -----------
        cortype: str, 
            The typle of parameters to cisualize their coreletions. Can be 
            ``num`` for numerical features and ``cat`` for categorical features. 
            *Default* is ``num`` for quantitative values. 
            
        method: str,  
            the correlation method. can be 'spearman' or `person`. *Default is
            ``pearson``
            
        features: List, optional 
            list of  the name of features for correlation analysis. If given, 
            must be sure that the names belong to the dataframe columns,  
            otherwise an error will occur. If features are valid, dataframe 
            is shrunk to the number of features before the correlation plot.
       
        min_periods: 
                Minimum number of observations required per pair of columns
                to have a valid result. Currently only available for 
                ``pearson`` and ``spearman`` correlation. For more details 
                refer to https://www.geeksforgeeks.org/python-pandas-dataframe-corr/
        
        sns_kws: Other seabon heatmap arguments. Refer to 
                https://seaborn.pydata.org/generated/seaborn.heatmap.html
                
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`EasyPlotter` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `EasyPlotter` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
             
           
        Example 
        ---------
        >>> from gofast.plot.explore  import EasyPlotter 
        >>> from gofast.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qplotObj = EasyPlotter().fit(data)
        >>> sns_kwargs ={'annot': False, 
        ...            'linewidth': .5, 
        ...            'center':0 , 
        ...            # 'cmap':'jet_r', 
        ...            'cbar':True}
        >>> qplotObj.plotCorrMatrix(cortype='cat', **sns_kwargs)
            
        """
        self.inspect 
        corc = str(copy.deepcopy(cortype))
        cortype= str(cortype).lower().strip() 
        if cortype.find('num')>=0 or cortype in (
                'value', 'digit', 'quan', 'quantitative'): 
            cortype ='num'
        elif cortype.find('cat')>=0 or cortype in (
                'string', 'letter', 'qual', 'qualitative'): 
            cortype ='cat'
        if cortype not in ('num', 'cat'): 
            return ValueError ("Expect 'num' or 'cat' for numerical and"
                               f" categorical features, not : {corc!r}")
       
        df_= self.data.copy(deep=True)
        # df_.reset_index(inplace=True )
        
        df_ = select_features(df_, features = features , 
                             include= 'number' if cortype  =='num' else None, 
                             exclude ='number' if cortype=='cat' else None,
                             )
        features = list(df_.columns ) # for consistency  

        if cortype =='cat': 
            for ftn in features: 
                df_[ftn] = df_[ftn].astype('category').cat.codes 
        
        elif cortype =='num': 
           
            if 'id' in features: 
                features.remove('id')
                df_= df_.drop('id', axis=1)

        ax= sns.heatmap(data =df_[list(features)].corr(
            method= method, min_periods=min_periods), 
                **sns_kws
                )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.fig_title)
        
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        plt.show()  if self.savefig is None else plt.close() 
        
        print(" --> Correlation matrix plot successfully done !" 
              ) if self.verbose > 0 else print()
              
        return self 
    
              
    def plotNumFeatures(
        self, 
        features=None, 
        coerce: bool= False,  
        map_lower_kws=None,
        **sns_kws): 
        """
        Plots qualitative features distribution using correlative aspect. Be 
        sure to provide numerical features as data arguments. 
        
        Parameters 
        -----------
        features: list
            List of numerical features to plot for correlating analyses. 
            will raise an error if features does not exist in the data 
            
        coerce: bool, 
            Constraint the data to read all features and keep only the numerical
            values. An error occurs if ``False`` and the data contains some 
            non-numericalfeatures. *default* is ``False``. 
            
        map_lower_kws: dict, Optional
            a way to customize plot. Is a dictionnary of sns.pairplot map_lower
            kwargs arguments. If the diagram `kind` is ``kde``, plot is customized 
            with the provided `map_lower_kws` arguments. if ``None``, 
            will check whether the `diag_kind` argument on `sns_kws` is ``kde``
            before triggering the plotting map. 
            
        sns_kws: dict, 
            Keywords word arguments of seabon pairplots. Refer to 
            http://seaborn.pydata.org/generated/seaborn.pairplot.html for 
            further details.             
                     
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`EasyPlotter` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `EasyPlotter` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
            
              
        Examples
        ---------
        >>> from gofast.plot.explore  import EasyPlotter 
        >>> from gofast.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qkObj = EasyPlotter(mapflow =False, target_name='flow'
                                  ).fit(data)
        >>> qkObj.sns_style ='darkgrid', 
        >>> qkObj.fig_title='Quantitative features correlation'
        >>> sns_pkws={'aspect':2 , 
        ...          "height": 2, 
        # ...          'markers':['o', 'x', 'D', 'H', 's',
        #                         '^', '+', 'S'],
        ...          'diag_kind':'kde', 
        ...          'corner':False,
        ...          }
        >>> marklow = {'level':4, 
        ...          'color':".2"}
        >>> qkObj.plotNumFeatures(coerce=True, map_lower_kws=marklow, **sns_pkws)
                                                
        """
        self.inspect 
            
        df_= self.data.copy(deep=True)
        
        try : 
            df_= df_.astype(float) 
        except: 
            if not coerce:
                non_num = list(select_features(df_, exclude='number').columns)
                msg = f"non-numerical features detected: {smart_format(non_num)}"
                warnings.warn(msg + "set 'coerce' to 'True' to only visualize"
                              " the numerical features.")
                raise ValueError (msg + "; set 'coerce'to 'True' to keep the"
                                  " the numerical insights")

        df_= select_features(df_, include ='number')
        ax =sns.pairplot(data =df_, hue=self.target_name,**sns_kws)
        
        if map_lower_kws is not None : 
            try : 
                sns_kws['diag_kind']
         
            except: 
                self._logging.info('Impossible to set `map_lower_kws`.')
                warnings.warn(
                    '``kde|sns.kdeplot``is not found for seaborn pairplot.'
                    "Impossible to lowering the distribution map.")
            else: 
                if sns_kws['diag_kind']=='kde' : 
                    ax.map_lower(sns.kdeplot, **map_lower_kws)
                    
        if self.savefig is not None :
            plt.savefig(self.savefig, dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        plt.show() if self.savefig is None else plt.close () 
        
        return self 
    
    def plotJoint2Features(
        self,
        features: List [str], *,
        join_kws=None, marginals_kws=None, 
        **sns_kws):
        """
        Joint method allows to visualize correlation of two features. 
        
        Draw a plot of two features with bivariate and univariate graphs. 
        
        Parameters 
        -----------
        features: list
            List of numerical features to plot for correlating analyses. 
            will raise an error if features does not exist in the data 

        join_kws:dict, optional 
            Additional keyword arguments are passed to the function used 
            to draw the plot on the joint Axes, superseding items in the 
            `joint_kws` dictionary.
            
        marginals_kws: dict, optional 
            Additional keyword arguments are passed to the function used 
            to draw the plot on the marginals Axes. 
            
        sns_kwargs: dict, optional
            keywords arguments of seaborn joinplot methods. Refer to 
            :ref:`<http://seaborn.pydata.org/generated/seaborn.jointplot.html>` 
            for more details about usefull kwargs to customize plots. 
          
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`EasyPlotter` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `EasyPlotter` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
              
             
        Examples
        ----------
        >>> from gofast.plot.explore  import EasyPlotter 
        >>> from gofast.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qkObj = EasyPlotter( lc='b', sns_style ='darkgrid', 
        ...             fig_title='Quantitative features correlation'
        ...             ).fit(data)  
        >>> sns_pkws={
        ...            'kind':'reg' , #'kde', 'hex'
        ...            # "hue": 'flow', 
        ...               }
        >>> joinpl_kws={"color": "r", 
                        'zorder':0, 'levels':6}
        >>> plmarg_kws={'color':"r", 'height':-.15, 'clip_on':False}           
        >>> qkObj.plotJoint2Features(features=['ohmS', 'lwi'], 
        ...            join_kws=joinpl_kws, marginals_kws=plmarg_kws, 
        ...            **sns_pkws, 
        ...            )
        """
        self.inspect 
  
        df_= self.data.copy(deep=True)
        
        if isinstance (features, str): 
            features =[features]
            
        if features is None: 
            self._logging.error(f"Valid features are {smart_format(df_.columns)}")
            raise PlotError("NoneType can not be a feature nor plotted.")
            
        df_= select_features(df_, features)

        # checker whether features is quantitative features 
        df_ = select_features(df_, include= 'number') 
        
        if len(df_.columns) != 2: 
            raise PlotError(f" Joinplot needs two features. {len(df_.columns)}"
                            f" {'was' if len(df_.columns)<=1 else 'were'} given")
            
            
        ax= sns.jointplot(data=df_, x=features[0], y=features[1],   **sns_kws)

        if join_kws is not None:
            join_kws = _assert_all_types(join_kws,dict)
            ax.plot_joint(sns.kdeplot, **join_kws)
            
        if marginals_kws is not None: 
            marginals_kws= _assert_all_types(marginals_kws,dict)
            
            ax.plot_marginals(sns.rugplot, **marginals_kws)
            
        plt.show() if self.savefig is None else plt.close () 
        
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        return self 
          
    def plotSemanticScatter(
        self,features: List [str], 
        *,
        relplot_kws= None, 
        **sns_kws 
        ): 
        """
        Draw a scatter plot with possibility of several semantic features 
        groupings.
        
        Indeed `plotSemanticScatter` analysis is a process of understanding 
        how features in a dataset relate to each other and how those
        relationships depend on other features. Visualization can be a core 
        component of this process because, when data are visualized properly,
        the human visual system can see trends and patterns that indicate a 
        relationship. 
        
        Parameters 
        -----------
        features: list
            List of numerical features to plot for correlating analyses. 
            will raise an error if features does not exist in the data 
            
        relplot_kws: dict, optional 
            Extra keyword arguments to show the relationship between 
            two features with semantic mappings of subsets.
            refer to :ref:`<http://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot>`
            for more details. 
            
        sns_kwargs:dict, optional
            kwywords arguments to control what visual semantics are used 
            to identify the different subsets. For more details, please consult
            :ref:`<http://seaborn.pydata.org/generated/seaborn.scatterplot.html>`. 
            
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        
        Returns
        -------
        :class:`EasyPlotter` instance
            Returns ``self`` for easy method chaining.
          
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `EasyPlotter` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
            
        Examples
        ----------
        >>> from gofast.plot.explore  import  EasyPlotter 
        >>> from gofast.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qkObj = EasyPlotter(lc='b', sns_style ='darkgrid', 
        ...             fig_title='geol vs lewel of water inflow',
        ...             xlabel='Level of water inflow (lwi)', 
        ...             ylabel='Flow rate in m3/h'
        ...            ) 
        >>>
        >>> qkObj.target_name='flow' # target the DC-flow rate prediction dataset
        >>> qkObj.mapflow=True  # to hold category FR0, FR1 etc..
        >>> qkObj.fit(data) 
        >>> marker_list= ['o','s','P', 'H']
        >>> markers_dict = {key:mv for key, mv in zip( list (
        ...                       dict(qkObj.data ['geol'].value_counts(
        ...                           normalize=True)).keys()), 
        ...                            marker_list)}
        >>> sns_pkws={'markers':markers_dict, 
        ...          'sizes':(20, 200),
        ...          "hue":'geol', 
        ...          'style':'geol',
        ...         "palette":'deep',
        ...          'legend':'full',
        ...          # "hue_norm":(0,7)
        ...            }
        >>> regpl_kws = {'col':'flow', 
        ...             'hue':'lwi', 
        ...             'style':'geol',
        ...             'kind':'scatter'
        ...            }
        >>> qkObj.plotSemanticScatter(features=['lwi', 'flow'],
        ...                         relplot_kws=regpl_kws,
        ...                         **sns_pkws, 
        ...                    )
            
        """
        self.inspect 
            
        df_= self.data.copy(deep=True)
        
        # controller function
        if isinstance (features, str): 
            features =[features]
            
        if features is None: 
            self._logging.error(f"Valid features are {smart_format(df_.columns)}")
            raise PlotError("NoneType can not be a feature nor plotted.")
            
        if len(features) < 2: 
            raise PlotError(f" Scatterplot needs at least two features. {len(df_.columns)}"
                            f" {'was' if len(df_.columns)<=1 else 'were'} given")
            
        # assert wether the feature exists 
        select_features(df_, features)
        
        ax= sns.scatterplot(data=df_,  x=features[0], y=features[1],
                              **sns_kws)
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.fig_title)
        
        if relplot_kws is not None: 
            relplot_kws = _assert_all_types(relplot_kws, dict)
            sns.relplot(data=df_, x= features[0], y=features[1],
                        **relplot_kws)
            
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        plt.show() if self.savefig is None else plt.close ()    
        
        return self 
       
    def plotDiscussingFeatures(
        self, features, *, 
        map_kws: Optional[dict]=None, 
        map_func: Optional[_F] = None, 
        **sns_kws)-> None: 
        """
        Provides the features names at least 04 and discuss with 
        their distribution. 
        
        This method maps a dataset onto multiple axes arrayed in a grid of
        rows and columns that correspond to levels of features in the dataset. 
        The plots produced are often called "lattice", "trellis", or
        'small-multiple' graphics. 
        
        Parameters 
        -----------
        features: list
            List of features for discussing. The number of recommended 
            features for better analysis is four (04) classified as below: 
                
                features_disposal = ['x', 'y', 'col', 'target|hue']
                
            where: 
                - `x` is the features hold to the x-axis, *default* is``ohmS`` 
                - `y` is the feature located on y_xis, *default* is ``sfi`` 
                - `col` is the feature on column subset, *default` is ``col`` 
                - `target` or `hue` for targetted examples, *default* is ``flow``
            
            If 03 `features` are given, the latter is considered as a `target`
        
       

        map_kws:dict, optional 
            Extra keyword arguments for mapping plot.
            
        func_map: callable, Optional 
            callable object,  is a plot style function. Can be a 'matplotlib-pyplot'
            function  like ``plt.scatter`` or 'seaborn-scatterplot' like 
            ``sns.scatterplot``. The *default* is ``sns.scatterplot``.
  
        sns_kwargs: dict, optional
           kwywords arguments to control what visual semantics are used 
           to identify the different subsets. For more details, please consult
           :ref:`<http://seaborn.pydata.org/generated/seaborn.FacetGrid.html>`.
           
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`EasyPlotter` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `EasyPlotter` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 

        Examples
        --------
        >>> from gofast.plot.explore  import  EasyPlotter 
        >>> from gofast.datasets import load_bagoue 
        >>> data = load_bagoue ().frame 
        >>> qkObj = EasyPlotter(  leg_kws={'loc':'upper right'},
        ...          fig_title = '`sfi` vs`ohmS|`geol`',
        ...            ) 
        >>> qkObj.target_name='flow' # target the DC-flow rate prediction dataset
        >>> qkObj.mapflow=True  # to hold category FR0, FR1 etc..
        >>> qkObj.fit(data) 
        >>> sns_pkws={'aspect':2 , 
        ...          "height": 2, 
        ...                  }
        >>> map_kws={'edgecolor':"w"}   
        >>> qkObj.plotDiscussingFeatures(features =['ohmS', 'sfi','geol', 'flow'],
        ...                           map_kws=map_kws,  **sns_pkws
        ...                         )   
        """
        self.inspect 

        df_= self.data.copy(deep=True)
        
        if isinstance (features, str ): 
            features =[features]
            
        if len(features)>4: 
            if self.verbose:  
                self._logging.debug(
                    'Features length provided is = {0:02}. The first four '
                    'features `{1}` are used for joinplot.'.format(
                        len(features), features[:4]))
                
            features=list(features)[:4]
            
        elif len(features)<=2: 
            if len(features)==2:verb, pl='are','s'
            else:verb, pl='is',''
            
            if self.verbose: 
                self._logging.error(
                    'Expect three features at least. {0} '
                    '{1} given.'.format(len(features), verb))
            
            raise PlotError(
                '{0:02} feature{1} {2} given. Expect at least 03 '
                'features!'.format(len(features),pl,  verb))
            
        elif len(features)==3:
            
            msg='03 Features are given. The last feature `{}` should be'\
                ' considered as the`targetted`feature or `hue` value.'.format(
                    features[-1])
            if self.verbose: 
                self._logging.debug(msg)
            
                warnings.warn(
                    '03 features are given, the last one `{}` is used as '
                    'target!'.format(features[-1]))
            
            features.insert(2, None)
    
        ax= sns.FacetGrid(data=df_, col=features[-2], hue= features[-1], 
                            **sns_kws)
        
        if map_func is None: 
            map_func = sns.scatterplot #plt.scatter
            
        if map_func is not None : 
            if not hasattr(map_func, '__call__'): 
                raise TypeError(
                    f'map_func must be a callable object not {map_func.__name__!r}'
                    )

        if map_kws is None : 
            map_kws = _assert_all_types(map_kws,dict)
            map_kws={'edgecolor':"w"}
            
        if (map_func and map_kws) is not None: 
            ax.map(map_func, features[0], features[1], 
                   **map_kws).add_legend(**self.leg_kws) 
      

        if self.savefig is not None :
            plt.savefig(self.savefig, dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
        
        plt.show() if self.savefig is None else plt.close ()
         
        return self 
         
    def PlotCoordDistributions(
        self,
        x:str =None, 
        y:str =None, 
        kind:str ='scatter',
        s_col ='lwi', 
        leg_kws:dict ={}, 
        **pd_kws
        ):
        """ Creates a plot  to visualize the sample distributions 
        according to the geographical coordinates `x` and `y`.
        
        Parameters 
        -----------
        x: str , 
            Column name to hold the x-axis values 
        y: str, 
            column na me to hold the y-axis values 
        s_col: column for scatter points. 'Default is ``fs`` time the features
            column `lwi`.
            
        pd_kws: dict, optional, 
            Pandas plot keywords arguments 
            
        leg_kws:dict, kws 
            Matplotlib legend keywords arguments 
           
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`EasyPlotter` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`EasyPlotter` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `EasyPlotter` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
        
        Examples
        --------- 
        >>> from gofast.transformers import StratifiedWithCategoryAdder
        >>> from gofast.plot.explore  import EasyPlotter
        >>> from gofast.datasets import load_bagoue 
        >>> df = load_bagoue ().frame
        >>> stratifiedNumObj= StratifiedWithCategoryAdder('flow')
        >>> strat_train_set , *_= \
        ...    stratifiedNumObj.fit_transform(X=df) 
        >>> pd_kws ={'alpha': 0.4, 
        ...         'label': 'flow m3/h', 
        ...         'c':'flow', 
        ...         'cmap':plt.get_cmap('jet'), 
        ...         'colorbar':True}
        >>> qkObj=EasyPlotter(fs=25.)
        >>> qkObj.fit(strat_train_set)
        >>> qkObj.PlotCoordDistributions( x= 'east', y='north', **pd_kws)
    
        """
        self.inspect
            
        df_= self.data.copy(deep=True)
        
         # visualize the data and get insights
        if 's' not in pd_kws.keys(): 
            pd_kws['s'] = df_[s_col]* self.fs 
             
        df_.plot(kind=kind, x=x, y=y, **pd_kws)
        
        self.leg_kws = self.leg_kws or dict () 
        
        plt.legend(**leg_kws)
        
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        plt.show () if self.savefig is None else plt.close()
        
        return self 
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, skip ='y') 
    
       
    def __getattr__(self, name):
        if not name.endswith ('__') and name.endswith ('_'): 
            raise NotFittedError (
                f"{self.__class__.__name__!r} instance is not fitted yet."
                " Call 'fit' method with appropriate arguments before"
               f" retreiving the attribute {name!r} value."
                )
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )      
       
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'data_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1
     
      
def viewtemplate (y, /, xlabel=None, ylabel =None,  **kws):
    """
    Quick view template
    
    Parameters 
    -----------
    y: Arraylike , shape (N, )
    xlabel: str, Optional 
        Label for naming the x-abscissia 
    ylabel: str, Optional, 
        Label for naming the y-coordinates.
    kws: dict, 
        keywords argument passed to :func:`matplotlib.pyplot.plot`

    """
    label =kws.pop('label', None)
    # create figure obj 
    obj = QuestPlotter()
    fig = plt.figure(figsize = obj.fig_size)
    ax = fig.add_subplot(1,1,1)
    ax.plot(y,
            color= obj.lc, 
            linewidth = obj.lw,
            linestyle = obj.ls , 
            label =label, 
            **kws
            )
    
    if obj.xlabel is None: 
        obj.xlabel =xlabel or ''
    if obj.ylabel is None: 
        obj.ylabel =ylabel  or ''

    ax.set_xlabel( obj.xlabel,
                  fontsize= .5 * obj.font_size * obj.fs 
                  )
    ax.set_ylabel (obj.ylabel,
                   fontsize= .5 * obj.font_size * obj.fs
                   )
    ax.tick_params(axis='both', 
                   labelsize=.5 * obj.font_size * obj.fs
                   )
    
    if obj.show_grid is True : 
        if obj.gwhich =='minor': 
              ax.minorticks_on() 
        ax.grid(obj.show_grid,
                axis=obj.gaxis,
                which = obj.gwhich, 
                color = obj.gc,
                linestyle=obj.gls,
                linewidth=obj.glw, 
                alpha = obj.galpha
                )
          
        if len(obj.leg_kws) ==0 or 'loc' not in obj.leg_kws.keys():
             obj.leg_kws['loc']='upper left'
        
        ax.legend(**obj.leg_kws)
        

        plt.show()
        
        if obj.savefig is not None :
            plt.savefig(obj.savefig,
                        dpi=obj.fig_dpi,
                        orientation =obj.fig_orientation
                        )     
     
        
        
        