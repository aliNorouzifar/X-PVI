from dash import html, dcc
import dash_bootstrap_components as dbc


layout = dbc.Container(
    className="page-container",
    children=[
        # Navigation Links
        html.Div(
            className="nav-links",
            children=[
                dcc.Link("Introduction", href="/", className="nav-link"),
                dcc.Link("X-PVI", href="/main", className="nav-link"),
                dcc.Link("About Me", href="/about_me", className="nav-link"),
            ],
        ),
        # Tool Name and Description
        html.Div(
            className="tool-name-container",
            children=[
                html.H1("Process Variant Identification", className="tool-name"),
                html.P(
                    "A cutting-edge tool for detecting and understanding process variability across performance dimensions.",
                    className="tool-subtitle",
                ),
            ],
        ),
        # Introduction Section
        html.Div(
            className="intro-section",
            children=[
                html.H2("Welcome to X-PVI!", className="section-title"),
                html.P(
                    "Processes often exhibit significant variability, posing challenges for process discovery and insight extraction. X-PVI helps you identify and explain these variations, empowering data-driven decision-making.",
                    className="content",
                ),
                dcc.Link(
                    "Get Started",
                    href="/main",
                    className="cta-link"
                ),
            ],
        ),
        # Background and Context Section
        html.Div(
            className="context-section",
            children=[
                html.H2("Background and Context", className="section-title"),
                html.P(
                    "This tool is based on a novel framework introduced in a recent study that focuses on uncovering process variability across multiple dimensions, including control-flow changes, case durations, and performance metrics. The framework addresses key challenges in process discovery, including the following:",
                    className="content",
                ),
                html.Ul(
                    children=[
                        html.Li("Detecting behavioral shifts using sliding window analysis with Earth Mover's Distance."),
                        html.Li("Encoding event logs into a feature space defined by declarative constraints for enhanced interpretability."),
                        html.Li("Clustering features to identify and explain distinct behavioral patterns."),
                    ],
                    className="feature-list",
                ),
                html.P(
                    "The framework was validated using real-life event logs from the UWV employee insurance agency in the Netherlands, demonstrating its ability to uncover meaningful changes, explain process variability, and support data-driven decisions.",
                    className="content",
                ),
                html.P(
                    "X-PVI is implemented as an open-source tool to enable broader applicability in real-world scenarios.",
                    className="content",
                ),
            ],
        ),
        # Example Section
        html.Div(
            className="example-section",
            children=[
                html.H2("Try the Tool with an Example", className="section-title"),
                html.P(
                    "Download a sample event log generated from a BPMN model to test the tool's features.",
                    className="content",
                ),
                html.A(
                    "Download Sample File",
                    href="/assets/test.xes",
                    download="test.xes",
                    className="download-link",
                ),
                html.Img(
                    src="/assets/bpmn_test.png",
                    alt="BPMN Model",
                    className="example-image",
                ),
            ],
        ),
    ],
)