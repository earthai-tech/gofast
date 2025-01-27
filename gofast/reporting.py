# -*- coding: utf-8 -*-

"""
Reporting utilities for gofast library: A high-level class for
generating comprehensive model performance or optimization reports.
"""

import os
import webbrowser
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt

#XXX TODO: Templating 
# The report 
# this should support : 
    # from gofast.reporting.templates import ...
    # from gofast.reporting.plots import ...
    # etc.
    
__all__=['ReportGenerator']

class ReportGenerator:
    r"""
    A versatile report generator for the gofast library, enabling
    users to create rich, interpretable HTML-based documents
    summarizing model performance, optimization procedures,
    and other diagnostics. By default, it expects structured
    data (e.g., dictionary of metrics or analysis results)
    and can optionally include plots or other artifacts.

    The primary entry point is :meth:`generate`, which takes
    the report data and writes an HTML file (or other formats)
    to the user-specified location.

    Examples
    --------
    >>> from gofast.reporting import ReportGenerator
    >>> results = {
    ...     'model_name': 'RandomForest',
    ...     'accuracy': 0.92,
    ...     'params': {'n_estimators': 100, 'max_depth': 10}
    ... }
    >>> # Create a basic HTML report of model performance:
    >>> ReportGenerator.generate(
    ...     report_data=results,
    ...     output_file='output/model_optimization_report.html',
    ...     include_plots=True
    ... )
    # A new HTML file is generated, summarizing the metrics with
    # optional visual plots for deeper insight.

    See Also
    --------
    Some hypothetical references or other gofast reporting
    modules. For instance, if gofast has specialized hooks
    for performance metrics, table generation, or a templating
    engine, those can be integrated here.

    Notes
    -----
    This class can be extended or overridden to tailor the
    resulting report layout. Additional arguments or
    data structures in `report_data` can be processed to
    embed custom charts or textual explanations.
    """

    @classmethod
    def generate(
        cls,
        report_data: Dict[str, Any],
        output_file: str,
        include_plots: bool = True,
        open_after: bool = False
    ) -> None:
        r"""
        Generate and save a comprehensive report (in HTML
        or other formats) summarizing performance, metrics,
        or optimization results for the gofast library.

        Parameters
        ----------
        report_data : dict
            A dictionary containing relevant metrics, model
            parameters, or analysis outputs to be included
            in the report. For example:

            .. code-block:: python

               {
                   'model': 'RandomForest',
                   'accuracy': 0.92,
                   'confusion_matrix': [[50, 2], [3, 45]],
                   'plots': [...],
                   ...
               }
        output_file : str
            The path (including filename) where the report
            is saved. Typically an ``.html`` or similar
            extension is recommended.
        include_plots : bool, optional
            Whether to include any plots or visual aids
            in the final report if available. Default=True.
        open_after : bool, optional
            If ``True``, attempts to open the resulting
            report in a web browser after generation.

        Returns
        -------
        None
            The function writes out an HTML file to
            ``output_file`` and performs no return action.

        Raises
        ------
        FileNotFoundError
            If the output directory does not exist and
            cannot be created (or user lacks permissions).
        ImportError
            If required libraries for plotting or templating
            are missing and `include_plots` is True.

        Notes
        -----
        Internally, this method might use a templating engine
        (like Jinja2) or inlined HTML generation to produce a
        final document. Plots can be base64-encoded or saved
        as separate files. The dictionary keys in `report_data`
        can be used to drive section content, e.g., "metrics",
        "training_config", etc.

        Examples
        --------
        >>> from gofast.reporting import ReportGenerator
        >>> example_data = {
        ...     'model_name': 'RandomForest',
        ...     'accuracy': 0.89,
        ...     'params': {...},
        ...     # Potentially other details, confusion matrix, etc.
        ... }
        >>> ReportGenerator.generate(
        ...     report_data=example_data,
        ...     output_file='results/my_model_report.html',
        ...     include_plots=True,
        ...     open_after=True
        ... )
        # A new HTML file is created and opened in the browser.
        """
        # 1) Ensure directory exists
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # 2) Build an HTML (or any format) from report_data
        html_content = cls._create_html_content(
            data=report_data,
            include_plots=include_plots
        )

        # 3) Write the file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # 4) Optionally open in browser
        if open_after:
            webbrowser.open_new_tab(output_file)

    @classmethod
    def _create_html_content(
        cls,
        data: Dict[str, Any],
        include_plots: bool
    ) -> str:
        r"""
        Internal helper method to convert a dictionary of analysis
        data into a minimal HTML summary. In practice, this could
        be extended with templating, embedded plots, or advanced
        formatting.
    
        Parameters
        ----------
        data : dict
            Dictionary containing analysis results or model
            metrics. The keys correspond to sections or item
            labels, and the values hold any textual or numeric
            details to be displayed.
    
        include_plots : bool
            Whether to include references to plots or images in
            the resulting HTML. If ``True``, this method inserts
            placeholder tags or real embedded content (as desired
            by the developer).
    
        Returns
        -------
        html : str
            A concatenated HTML document string. In more complex
            usage, this could be a partial snippet combined with
            a larger template or CSS layout.
    
        Notes
        -----
        This basic approach merely enumerates each key-value pair
        in ``data`` and wraps them in an HTML list item. If
        ``include_plots=True``, a placeholder paragraph is added
        to highlight where plot embedding logic might go.
    
        Examples
        --------
        >>> data_dict = {
        ...     'model_name': 'RandomForest',
        ...     'accuracy': 0.92,
        ...     'other_metrics': {'precision': 0.88, 'recall': 0.91}
        ... }
        >>> html_str = MyReportClass._create_html_content(
        ...     data_dict, include_plots=True
        ... )
        >>> # Then write `html_str` to file or combine in a
        ... # bigger template.
    
        .. math::
            \text{HTML} = \sum_{(k,v)\in data} \langle li\rangle
            \mathbf{k}: v \langle/li\rangle
    
        """
       #XXX TODO: Developping templates for Gofast. report html display. 
       
        # Start building the HTML
        lines = [
            "<html>",
            "<head><title>Gofast Report</title></head>",
            "<body>",
            "<h1>Gofast Model Report</h1>",
            "<ul>"
        ]
    
        # Enumerate each item in data as a bullet point
        for key, value in data.items():
            # For complex types (lists, dicts), str() them here
            lines.append(
                f"<li><strong>{key}:</strong> {value}</li>"
            )
    
        lines.append("</ul>")
    
        # Optionally add placeholders for plots/images
        if include_plots:
            lines.append(
                "<p>(Plot placeholders or real images "
                "could be embedded here.)</p>"
            )
    
        lines.append("</body></html>")
    
        # Join lines with newline separation
        html_content = "\n".join(lines)
        return html_content
