# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Reporting utilities for gofast library: A high-level class for
generating comprehensive model performance or optimization reports.
"""

import os
import webbrowser
from typing import Any, Dict, Optional, Union
import pandas as pd 

from .api.summary import ResultSummary, ModelSummary
from .api.summary import ReportFactory, Summary 
from .api.types import DataFrame

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
        report_data: Union [
            Dict[str, Any], 
            ResultSummary, Summary, ModelSummary, ReportFactory, DataFrame 
            ],
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
        data: Union [
            Dict[str, Any], 
            ResultSummary, Summary, ModelSummary, ReportFactory, DataFrame 
            ],
        include_plots: bool
    ) -> str:
        r"""
        Internal helper method to convert a dictionary of analysis
        data into a minimal HTML summary. In practice, this could
        be extended with templating, embedded plots, or advanced
        formatting.
        
        This method attempts to render the report content using one of the
        specialized rendering functions if ``data`` is an instance of a known
        summary/report class. Specifically, it checks for instances of
        :class:`ResultSummary`, :class:`ModelSummary`, :class:`ReportFactory`,
        :class:`Summary`, or a pandas DataFrame, and then delegates rendering
        to the corresponding function (e.g., :func:`render_result_summary_html`,
        :func:`render_model_summary_html`, :func:`render_report_factory_html`,
        :func:`render_summary_html`, or :func:`render_dataframe_html`). If
        ``data`` does not match any of these types, it falls back to a minimal
        HTML summary generated directly from the dictionary contents.
    
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
        
        >>> from gofast.reporting import ReportFactory, render_report_factory_html
        >>> report_data = {'model_name': 'RandomForest', 'accuracy': 0.92}
        >>> report = ReportFactory(title="Model Report")
        >>> report.add_mixed_types(report_data)
        >>> html_content = ReportFactory._create_html_content(report, 
        ...                    include_plots=True)
        >>> print(html_content)
        
    
        .. math::
            \text{HTML} = \sum_{(k,v)\in data} \langle li\rangle
            \mathbf{k}: v \langle/li\rangle
    
        """
        if isinstance (
            data, (Summary, ReportFactory, ModelSummary, ResultSummary, pd.DataFrame)
            ): 
            return _render_html_if(data)
        
        #XXX TODO: NEXT step is to develop a complete  templates for Gofast
        #       specifically. report html display. 
        #       Start building the HTML content as a list of lines.
        
        lines = [
            "<html>",
            "  <head>",
            "    <title>Gofast Report</title>",
            "    <meta charset='utf-8'>",
            "    <style>",
            "      body { font-family: Arial, sans-serif; margin: 20px; }",
            "      h1   { color: #333; }",
            "      ul   { list-style-type: none; padding: 0; }",
            "      li   { margin-bottom: 10px; }",
            "      .section-title { font-weight: bold; }",
            "    </style>",
            "  </head>",
            "  <body>",
            "    <h1>Gofast Model Report</h1>",
            "    <hr>",
            "    <ul>"
        ]

        # Iterate over the key-value pairs in the data dictionary.
        for key, value in data.items():
            # Convert complex objects (like dicts or lists) to a
            # preformatted string for better display.
            if isinstance(value, (dict, list)):
                display_value = f"<pre>{str(value)}</pre>"
            else:
                display_value = str(value)
            # Append each item as an HTML list item.
            lines.append(
                f"      <li><span class='section-title'>{key}:</span> "
                f"{display_value}</li>"
            )

        # Close the unordered list.
        lines.append("    </ul>")

        # Optionally insert a placeholder for plots/images.
        if include_plots:
            # Check if the input data dictionary contains a 'plots' key.
            # If it does, assume it is a list of base64-encoded PNG images
            # or valid image URLs and embed each image into the report.
            plots = data.get('plots', None)
            if plots:
                lines.append("    <div class='plots'>")
                lines.append("      <h2>Visualizations</h2>")
                for idx, img in enumerate(plots):
                    # If img is a base64 string, it should 
                    # already include the data URI prefix,
                    # or you can prepend "data:image/png;base64," if needed.
                    lines.append(
                        f"      <div class='plot'>"
                        f"        <img src='{img}' alt='Plot {idx+1}' "
                        f"style='max-width: 100%; height: auto;'/>"
                        f"      </div>"
                    )
                lines.append("    </div>")
            else:
                # Otherwise, add a default  message.
                lines.append(
                    "    <div class='plots'>"
                    "      <p>(No plots available at this time. "
                    "Plot placeholders or embedded images could be added here.)</p>"
                    "    </div>"
                )

        # Append closing tags for the body and HTML document.
        lines.append("  </body>")
        lines.append("</html>")

        # Combine the list of lines into a single HTML string.
        html_content = "\n".join(lines)
        return html_content
    
def _render_html_if(data) -> str:
    r"""
    Render the input data into an HTML string using specialized
    rendering functions when possible. If the input object is an instance
    of a known summary/report class (e.g., :class:`ResultSummary`, 
    :class:`ModelSummary`, :class:`ReportFactory`, :class:`Summary`, or a 
    pandas DataFrame), the corresponding rendering function is used. If the 
    data type is not recognized, a fallback minimal HTML summary is generated
    by iterating over the dictionary key-value pairs.

    Returns
    -------
    str
        A complete HTML document as a string.
    """
    # Use specialized rendering if data matches known classes.
    if isinstance(data, ResultSummary):
        return render_result_summary_html(data)
    elif isinstance(data, ModelSummary):
        return render_model_summary_html(data)
    elif isinstance(data, ReportFactory):
        return render_report_factory_html(data)
    elif isinstance(data, pd.DataFrame):
        return render_dataframe_html(data, title="Data Report")
    elif isinstance(data, Summary):
        return render_summary_html(data)


def render_result_summary_html(summary: ResultSummary) -> str:
    r"""
    Generate an HTML representation of a ResultSummary object.

    This helper function recursively converts the
    :obj:`results` attribute of a ResultSummary instance into
    a structured HTML fragment. Nested dictionaries are rendered
    as nested unordered lists to provide a clear hierarchical
    visualization of the summary data.

    Mathematically, if :math:`R` represents the result dictionary,
    the function produces HTML equivalent to:

    .. math::
       \text{HTML} = \sum_{(k,v) \in R} \langle li \rangle \,
       k : v \, \langle /li \rangle

    Parameters
    ----------
    `summary`     : ResultSummary
        An instance of the ResultSummary class that contains the
        results (stored in the ``results`` attribute) to be rendered.
        This object may include nested dictionaries representing
        complex data structures.

    Returns
    -------
    str
        A string containing HTML that represents the result summary.
        The HTML is structured with a header displaying the summary
        title and a nested list for the results.

    Examples
    --------
    >>> from gofast.api.summary import ResultSummary
    >>> summary = ResultSummary(name="Data Check", pad_keys="auto", 
    ...                          max_char=50)
    >>> summary.add_results({
    ...     'Accuracy': 0.92,
    ...     'Confusion Matrix': {'TP': 50, 'FP': 5, 'FN': 3, 'TN': 42}
    ... })
    >>> html_str = render_result_summary_html(summary)
    >>> print(html_str)
    <div class='result-summary'>
      <h2>Data Check</h2>
      <ul>
        <li><strong>Accuracy:</strong> 0.92</li>
        <li><strong>Confusion Matrix:</strong>
          <ul>
            <li><strong>TP:</strong> 50</li>
            <li><strong>FP:</strong> 5</li>
            <li><strong>FN:</strong> 3</li>
            <li><strong>TN:</strong> 42</li>
          </ul>
        </li>
      </ul>
    </div>

    Notes
    -----
    This function can be used as a helper within the
    :meth:`ReportGenerator._create_html_content` method to embed
    detailed result summaries in generated reports. If the
    summary data is very large or complex, consider adjusting the
    formatting parameters (such as key padding or maximum character
    limits) in the ResultSummary instance prior to rendering.

    See Also
    --------
    ReportGenerator.generate : Main method to generate HTML reports.
    ResultSummary.__str__    : Provides a text-based summary of the results.

    References
    ----------
    .. [1] Wickham, H. (2014). "Tidy Data". Journal of Statistical
           Software, 59(10), 1-23.
    """
    def render_dict(d: dict) -> str:
        """
        Recursively converts a dictionary to an HTML unordered list.

        Parameters
        ----------
        `d` : dict
            The dictionary to render.

        Returns
        -------
        str
            HTML string representing the dictionary.
        """
        lines = ["<ul>"]
        for key, value in d.items():
            # If value is a nested dictionary, recursively render it.
            if isinstance(value, dict):
                rendered_value = render_dict(value)
                lines.append(
                    f"<li><strong>{key}:</strong> {rendered_value}</li>"
                )
            else:
                lines.append(
                    f"<li><strong>{key}:</strong> {value}</li>"
                )
        lines.append("</ul>")
        return "\n".join(lines)

    html_lines = [
        "<div class='result-summary'>",
        f"  <h2>{summary.name}</h2>",
        render_dict(summary.results),
        "</div>"
    ]
    return "\n".join(html_lines)

def render_model_summary_html(
    summary: ModelSummary
) -> str:
    r"""
    Render a ModelSummary instance as a structured HTML document.

    This helper function takes a ModelSummary object—which holds
    a formatted summary report and associated metadata—and returns
    an HTML string that visually presents the model tuning results.
    The HTML output includes a header with the summary title and a
    preformatted block containing the summary report.

    Mathematically, if the summary report is represented as a string
    :math:`S`, then the HTML is given by:

    .. math::
       \text{HTML} = \text{header} \oplus \langle pre \rangle S 
       \langle /pre \rangle

    where :math:`\oplus` denotes concatenation of HTML elements.

    Parameters
    ----------
    `summary` : ModelSummary
        An instance of the ModelSummary class containing the
        summary report (in the attribute ``summary_report``) and
        a title (in the attribute ``title``).

    Returns
    -------
    str
        A complete HTML document as a string representing the
        model summary. If the summary report is empty, a message
        indicating that no summary is available is displayed.

    Examples
    --------
    >>> from gofast.api.summary import ModelSummary
    >>> summary = ModelSummary(title="SVC Performance")
    >>> # Assume summary.add_performance(model_results) was called
    >>> html_str = render_model_summary_html(summary)
    >>> print(html_str)
    <html>
      <head><title>SVC Performance</title></head>
      <body>
        <h1>SVC Performance</h1>
        <div class='summary-report'>
          <pre>... summary report content ...</pre>
        </div>
      </body>
    </html>

    Notes
    -----
    This function is intended to be used as a helper for generating
    HTML reports in the gofast reporting module. It assumes that the
    ModelSummary instance has already been populated with a summary
    report. For advanced formatting, this function can be extended
    or integrated with a templating engine.

    See Also
    --------
    ReportGenerator._create_html_content : Similar method for rendering
        result summaries.
    ResultSummary.__str__           : Provides text-based summary output.

    References
    ----------
    .. [1] Wickham, H. (2014). "Tidy Data". Journal of Statistical
           Software, 59(10), 1-23.
    """
    # Begin constructing the HTML document as a list of lines.
    html_lines = [
        "<html>",
        "  <head>",
        f"    <title>{summary.title if summary.title else 'Model Summary'}</title>",
        "    <meta charset='utf-8'>",
        "    <style>",
        "      body { font-family: Arial, sans-serif; margin: 20px; }",
        "      h1   { color: #333; }",
        "      .summary-report { background-color: #f9f9f9; ",
        "                        padding: 10px; border: 1px solid #ddd; }",
        "      pre { white-space: pre-wrap; word-wrap: break-word; }",
        "    </style>",
        "  </head>",
        "  <body>",
        f"    <h1>{summary.title if summary.title else 'Model Summary'}</h1>",
        "    <hr>",
        "    <div class='summary-report'>"
    ]

    # If summary_report is populated, embed it in a <pre> tag for formatting.
    if summary.summary_report:
        html_lines.append(f"      <pre>{summary.summary_report}</pre>")
    else:
        html_lines.append("      <p>No summary report available.</p>")

    # Close the div, body, and html tags.
    html_lines.extend([
        "    </div>",
        "  </body>",
        "</html>"
    ])

    # Combine the lines into a single HTML string.
    return "\n".join(html_lines)

def render_summary_html(summary: Summary) -> str:
    r"""
    Render a Summary instance as a structured HTML document.

    This helper function converts a Summary object—which encapsulates
    a detailed summary report (stored in the ``summary_report`` attribute)
    and associated metadata—into a formatted HTML document. The resulting
    HTML includes a header with the summary title and a preformatted section
    that displays the summary report in a readable manner.

    Mathematically, if :math:`S` denotes the summary report string, then
    the HTML output is given by:

    .. math::
       \text{HTML} = \langle html \rangle \oplus \langle head \rangle
       \oplus \langle body \rangle \oplus \langle pre \rangle S
       \langle /pre \rangle \oplus \langle /body \rangle \oplus
       \langle /html \rangle

    where :math:`\oplus` denotes string concatenation.

    Parameters
    ----------
    `summary` : Summary
        An instance of the Summary class containing the summary report to be
        rendered. It should have a non-empty ``summary_report`` attribute and
        an optional ``title`` attribute.

    Returns
    -------
    str
        A complete HTML document as a string representing the summary.

    Examples
    --------
    >>> from gofast.api.summary import Summary
    >>> summary = Summary(title="Employee Data Overview")
    >>> summary.add_basic_statistics(df)
    >>> html_output = render_summary_html(summary)
    >>> print(html_output)
    <html>
      <head>
        <title>Employee Data Overview</title>
        ...
      </head>
      <body>
        <h1>Employee Data Overview</h1>
        <div class="summary-report">
          <pre>... formatted summary report ...</pre>
        </div>
      </body>
    </html>

    Notes
    -----
    This function is intended as a helper for generating HTML-based reports
    from Summary objects. For more sophisticated rendering, consider
    integrating with a templating engine such as Jinja2.

    See Also
    --------
    render_model_summary_html : Render HTML for ModelSummary instances.
    ReportGenerator._create_html_content : Generate HTML reports for gofast.

    References
    ----------
    .. [1] Wickham, H. (2014). "Tidy Data". Journal of Statistical
           Software, 59(10), 1-23.
    """
    # Begin constructing the HTML content as a list of strings.
    html_lines = [
        "<html>",
        "  <head>",
        f"    <title>{summary.title if summary.title else 'Summary Report'}</title>",
        "    <meta charset='utf-8'>",
        "    <style>",
        "      body { font-family: Arial, sans-serif; margin: 20px; }",
        "      h1   { color: #333; }",
        "      .summary-report { background: #f5f5f5; padding: 10px; ",
        "                        border: 1px solid #ddd; }",
        "      pre { white-space: pre-wrap; word-wrap: break-word; }",
        "    </style>",
        "  </head>",
        "  <body>",
        f"    <h1>{summary.title if summary.title else 'Summary Report'}</h1>",
        "    <hr>",
        "    <div class='summary-report'>"
    ]

    # Embed the summary report in a preformatted block.
    if summary.summary_report:
        html_lines.append(f"      <pre>{summary.summary_report}</pre>")
    else:
        html_lines.append("      <p>No summary report available.</p>")

    # Optionally, add additional details from the Summary object's results.
    if hasattr(summary, "results") and summary.results:
        html_lines.append("    <h2>Detailed Results</h2>")
        html_lines.append("    <ul>")
        for key, value in summary.results.items():
            html_lines.append(
                f"      <li><strong>{key}:</strong> {value}</li>"
            )
        html_lines.append("    </ul>")

    # Close HTML tags.
    html_lines.extend([
        "    </div>",
        "  </body>",
        "</html>"
    ])

    # Join the list of lines into a single HTML string.
    return "\n".join(html_lines)

def render_report_factory_html(report: ReportFactory) -> str:
    r"""
    Generate a robust HTML representation of a ReportFactory object.

    This helper function converts a :class:`ReportFactory` instance,
    which contains report data in its ``report`` and 
    ``report_str`` attributes, into a structured HTML document.
    The HTML output includes a header with the report title, a section
    displaying the formatted report summary, and, if available, a
    detailed listing of raw report data. The generated HTML can be
    used directly for visualization in a browser or embedded in a web
    page.

    Mathematically, if :math:`S` represents the summary string and
    :math:`R` represents the raw report data, then the output is
    given by:

    .. math::
       \text{HTML} = \langle html \rangle \oplus \langle head \rangle 
       \oplus \langle body \rangle \oplus \langle h1 \rangle (S)
       \oplus \langle div \rangle (S) \oplus \langle div \rangle (R)
       \oplus \langle /body \rangle \oplus \langle /html \rangle

    Parameters
    ----------
    `report`       : ReportFactory
        A :class:`ReportFactory` instance containing the report to be
        rendered. Expected attributes include:
        
        - ``title``: a string title for the report.
        - ``report_str``: a formatted summary of the report.
        - ``report``: raw report data, typically a dictionary.

    Returns
    -------
    str
        A string containing a complete HTML document representing the
        report.

    Examples
    --------
    >>> from gofast.reporting import ReportFactory
    >>> report = ReportFactory(title="Sales Analysis")
    >>> report.add_mixed_types({'Total Sales': 123456.78,
    ...                          'Average Rating': 4.56})
    >>> html_output = render_report_factory_html(report)
    >>> print(html_output)
    <html>
      <head>
        <title>Sales Analysis</title>
        ...
      </head>
      <body>
        <h1>Sales Analysis</h1>
        <div class='report section'>
          <pre>... formatted report ...</pre>
        </div>
        <div class='raw-report section'>
          <h2>Raw Report Data</h2>
          <ul>
            <li><strong>Total Sales:</strong> 123456.78</li>
            <li><strong>Average Rating:</strong> 4.56</li>
          </ul>
        </div>
      </body>
    </html>

    See Also
    --------
    ReportGenerator._create_html_content : For generating HTML reports 
        from result summaries.
    render_result_summary_html            : For rendering ResultSummary 
        objects as HTML.

    References
    ----------
    .. [1] Wickham, H. (2014). "Tidy Data". Journal of Statistical
           Software, 59(10), 1-23.
    """
    # Initialize the HTML document as a list of lines.
    html_lines = [
        "<html>",
        "  <head>",
        f"    <title>{report.title or 'Report'}</title>",
        "    <meta charset='utf-8'>",
        "    <style>",
        "      body { font-family: Arial, sans-serif; margin: 20px; }",
        "      h1   { color: #333; }",
        "      .section { margin-bottom: 20px; }",
        "      .report { background: #f9f9f9; padding: 10px; ",
        "                border: 1px solid #ddd; }",
        "      pre { white-space: pre-wrap; word-wrap: break-word; }",
        "      ul { list-style-type: none; padding: 0; }",
        "      li { margin-bottom: 8px; }",
        "    </style>",
        "  </head>",
        "  <body>",
        f"    <h1>{report.title or 'Report'}</h1>",
        "    <hr>",
        "    <div class='report section'>"
    ]

    # Embed the formatted report summary if available.
    if report.report_str:
        html_lines.append(f"      <pre>{report.report_str}</pre>")
    else:
        html_lines.append("      <p>No report summary available.</p>")
    html_lines.append("    </div>")

    # If raw report data exists and is a dictionary, add a raw data section.
    if isinstance(report.report, dict) and report.report:
        html_lines.extend([
            "    <div class='raw-report section'>",
            "      <h2>Raw Report Data</h2>",
            "      <ul>"
        ])
        for key, value in report.report.items():
            html_lines.append(
                f"        <li><strong>{key}:</strong> {value}</li>"
            )
        html_lines.extend([
            "      </ul>",
            "    </div>"
        ])

    # Close the HTML document.
    html_lines.extend([
        "  </body>",
        "</html>"
    ])

    # Return the HTML as a single concatenated string.
    return "\n".join(html_lines)

def render_dataframe_html(
    df: pd.DataFrame,
    title: str = "DataFrame Report",
    css: Optional[str] = None,
    table_class: str = "gofast-table"
) -> str:
    r"""
    Render a pandas DataFrame as a beautifully styled HTML document
    suitable for gofast software reports.

    This function converts a DataFrame into an HTML table and wraps it
    in a complete HTML document. Custom CSS can be provided to override
    the default styling. The resulting document is designed to be both
    visually appealing and easy to read.

    .. math::
       \text{HTML} = \langle html \rangle \oplus \langle head \rangle 
       \oplus \langle body \rangle \oplus \langle h1 \rangle T
       \oplus \langle table \rangle \langle /body \rangle \oplus
       \langle /html \rangle

    where :math:`T` is the HTML table generated from the DataFrame.

    Parameters
    ----------
    `df`         : pandas.DataFrame
        The DataFrame to be rendered.
    `title`      : str, optional
        The title of the HTML document. This title is displayed at the
        top of the report. Default is ``"DataFrame Report"``.
    `css`        : str, optional
        Custom CSS styles for the HTML document. If not provided,
        a default style is used.
    `table_class`: str, optional
        The CSS class assigned to the HTML table generated from the
        DataFrame. Default is ``"gofast-table"``.

    Returns
    -------
    str
        A complete HTML document as a string representing the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6]
    ... })
    >>> html_doc = render_dataframe_html(df, title="My Data Report")
    >>> print(html_doc)

    Notes
    -----
    The default CSS ensures that the HTML document has a clean, modern
    look with a responsive table. The table features alternating row
    colors and hover effects for enhanced readability.

    See Also
    --------
    pandas.DataFrame.to_html : Method to convert a DataFrame to HTML.
    """
   

    if css is None:
        css = (
            "body { font-family: Arial, sans-serif; margin: 20px; "
            "background-color: #f4f4f4; } \n"
            "h1 { text-align: center; color: #333; } \n"
            f".{table_class} {{ width: 100%; border-collapse: collapse; "
            "margin-top: 20px; }} \n"
            f".{table_class} th, .{table_class} td {{ border: 1px solid #ccc; "
            "padding: 8px; text-align: center; }} \n"
            f".{table_class} th {{ background-color: #333; color: #fff; }} \n"
            f".{table_class} tr:nth-child(even) {{ background-color: #e9e9e9; }} \n"
            f".{table_class} tr:hover {{ background-color: #d0d0d0; }}"
        )

    # Generate HTML table from the DataFrame using pandas built-in method.
    table_html = df.to_html(classes=table_class, index=True, border=0)

    # Construct the full HTML document.
    html_doc = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "  <head>\n"
        "    <meta charset='utf-8'>\n"
        f"    <title>{title}</title>\n"
        "    <style>\n"
        f"{css}\n"
        "    </style>\n"
        "  </head>\n"
        "  <body>\n"
        f"    <h1>{title}</h1>\n"
        f"    {table_html}\n"
        "  </body>\n"
        "</html>"
    )

    return html_doc
