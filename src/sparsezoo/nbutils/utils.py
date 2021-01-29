# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code related to model repo selection display in a jupyter notebook using ipywidgets
"""

from typing import Dict, List, Tuple

import ipywidgets as widgets
from sparsezoo.objects import Model


__all__ = ["ModelSelectWidgetContainer", "SelectDomainWidgetContainer"]


def format_html(
    message: str, header: str = None, color: str = None, italic: bool = False
):
    """
    Create a message formatted as html using the given parameters.
    Expected to be used in ipywidgets.HTML.

    :param message: the message string to display
    :param header: the header type to use for the html (h1, h2, h3, h4, etc)
    :param color: the color to apply as a style (black, red, green, etc)
    :param italic: True to format the HTML as italic, False otherwise
    :return: the message formatted as html
    """
    if not message:
        message = ""

    message = "<i>{}</i>".format(message) if italic else message
    color = 'style="color: {}"'.format(color) if color else ""
    obj = "span" if not header else header
    html = "<{} {}>{}</{}>".format(obj, color, message, obj)

    return html


class SelectDomainWidgetContainer(object):
    """
    Widget used in model repo notebooks for selecting domain and subdomain to
        search within

    :param domains: list of domains to look through
    :param sub_domain: a map of sub domains for each domain
    """

    def __init__(
        self,
        domains: List[str] = ["cv"],
        sub_domains: Dict[str, List[str]] = {
            "cv": ["classification"],
        },
    ):
        self._domains_dropdown = widgets.Dropdown(options=domains, value=domains[0])
        self._sub_domains_dropdown = widgets.Dropdown(
            options=sub_domains[domains[0]], value=sub_domains[domains[0]][0]
        )
        self._domains = domains
        self._sub_domains = sub_domains

    def create(self):
        def _domain_selector(change):
            domain = change["new"]
            sub_domains = (
                self._sub_domains[domain] if domain in self._sub_domains else []
            )
            self._sub_domains_dropdown.options = sub_domains
            if self._sub_domains_dropdown.value not in sub_domains:
                self._sub_domains_dropdown.value = (
                    sub_domains[0] if len(sub_domains) > 0 else None
                )

        self._domains_dropdown.observe(_domain_selector, names="value")
        self.container = widgets.VBox(
            (
                widgets.HTML(format_html("Model Search Domains", header="h2")),
                widgets.HBox(
                    (widgets.HTML(format_html("Domains:")), self._domains_dropdown)
                ),
                widgets.HBox(
                    (
                        widgets.HTML(format_html("Sub Domains:")),
                        self._sub_domains_dropdown,
                    )
                ),
            )
        )
        return self.container

    @property
    def selected_domain_info(self) -> Tuple[str, str]:
        return (self._domains_dropdown.value, self._sub_domains_dropdown.value)


def _optimazation_id(mod: Model):
    return (
        "-".join([mod.optim_name, mod.optim_category])
        if mod.optim_target is None
        else "-".join([mod.optim_name, mod.optim_category, mod.optim_target])
    )


class _ModelsWidget(object):
    def __init__(self, all_models: List[Model]):
        self._all_models = all_models
        self._architecture_selector = widgets.Select(
            options=[], description="Networks:"
        )
        self._dataset_selector = widgets.Select(options=[], description="Dataset:")
        self._framework_selector = widgets.Select(
            options=[], description="ML Framework:"
        )
        self._opt_name_selector = widgets.Select(options=[], description="Type:")
        self._selected_text = widgets.HTML(format_html("", italic=True))

        self.container = widgets.VBox(
            (
                widgets.HTML(format_html("Selection", header="h4")),
                widgets.HBox(
                    (
                        self._architecture_selector,
                        self._dataset_selector,
                        self._framework_selector,
                        self._opt_name_selector,
                    )
                ),
                widgets.HBox(
                    (
                        widgets.HTML(format_html("Selected:", header="h5")),
                        self._selected_text,
                    )
                ),
            )
        )
        self._filtered: List[Model] = []
        self._setup_hooks()
        self.selected = None

    @property
    def selected_framework(self) -> str:
        return self._framework_selector.value

    def update(self, filtered: List[Model]):
        self._filtered = filtered
        self._update_selectors()

    def _setup_hooks(self):
        def _selector_change(change):
            if change["new"] != change["old"]:
                self._update_selectors()

        self._architecture_selector.observe(_selector_change, names="value")
        self._dataset_selector.observe(_selector_change, names="value")
        self._framework_selector.observe(_selector_change, names="value")
        self._opt_name_selector.observe(_selector_change, names="value")

    def _update_selectors(self):
        architecture = self._architecture_selector.value
        dataset = self._dataset_selector.value
        framework = self._framework_selector.value
        optimization_name = self._opt_name_selector.value

        architectures = {mod.architecture for mod in self._filtered}
        architectures = list(architectures)
        architectures.sort()
        if architecture not in architectures:
            architecture = architectures[0] if len(architectures) > 0 else None
        self._architecture_selector.options = architectures
        self._architecture_selector.value = architecture

        datasets = {
            mod.dataset for mod in self._filtered if mod.architecture == architecture
        }
        datasets = list(datasets)
        datasets.sort()
        if dataset not in datasets:
            dataset = datasets[0] if len(datasets) > 0 else None
        self._dataset_selector.options = datasets
        self._dataset_selector.value = dataset

        frameworks = {
            mod.framework
            for mod in self._filtered
            if mod.architecture == architecture and mod.dataset == dataset
        }
        frameworks = list(frameworks)
        frameworks.sort()
        if framework not in frameworks:
            framework = frameworks[0] if len(frameworks) > 0 else None
        self._framework_selector.options = frameworks
        self._framework_selector.value = framework

        opt_names = {
            _optimazation_id(mod)
            for mod in self._filtered
            if mod.architecture == architecture
            and mod.dataset == dataset
            and (mod.framework == framework)
        }
        opt_names = list(opt_names)
        opt_names.sort()
        if optimization_name not in opt_names:
            optimization_name = opt_names[0] if len(opt_names) > 0 else None
        self._opt_name_selector.options = opt_names
        self._opt_name_selector.value = optimization_name

        self._update_selected()

    def _update_selected(self):
        self.selected = None
        self._selected_text.value = ""

        mods = [
            mod
            for mod in self._filtered
            if (
                mod.architecture == self._architecture_selector.value
                and mod.dataset == self._dataset_selector.value
                and (mod.framework == self._framework_selector.value)
                and _optimazation_id(mod) == self._opt_name_selector.value
            )
        ]
        if len(mods) > 0:
            mod = mods[0]
            self.selected = mod
            self._selected_text.value = format_html(mod.display_name, italic=True)


class _FilterWidget(object):
    def __init__(self, all_models: List[Model]):
        self._all_models = all_models
        self._datasets = self._init_datasets()
        self._repos = self._init_repos()

        self._recal_checkbox = widgets.Checkbox(value=False, indent=False)
        self._datasets_dropdown = widgets.Dropdown(
            options=self._datasets, value=self._datasets[0]
        )
        self._repo_dropdown = widgets.Dropdown(
            options=self._repos, value=self._repos[0]
        )
        self._setup_hooks()

        self.container = widgets.VBox(
            (
                widgets.HTML(format_html("Filters", header="h4")),
                widgets.HBox(
                    (widgets.HTML(format_html("Datasets:")), self._datasets_dropdown)
                ),
                widgets.HBox(
                    (widgets.HTML(format_html("Repos:")), self._repo_dropdown)
                ),
                widgets.HBox(
                    (
                        widgets.HTML(format_html("Recalibrated Only:")),
                        self._recal_checkbox,
                    )
                ),
            )
        )
        self.filtered_callback = None

    def _init_datasets(self):
        datasets = [mod.dataset for mod in self._all_models]
        datasets = list(dict.fromkeys(datasets))
        datasets.sort()
        return ["all datasets"] + datasets

    def _init_repos(self):
        repos = [mod.repo for mod in self._all_models]
        repos = list(dict.fromkeys(repos))
        repos.sort()
        return ["all repos"] + repos

    def _setup_hooks(self):
        def _invoke_callback():
            datasets = (
                [self._datasets_dropdown.value]
                if self._datasets_dropdown.value != "all datasets"
                and self._datasets_dropdown.value
                else None
            )
            repos = (
                [self._repo_dropdown.value]
                if self._repo_dropdown.value != "all repos"
                and self._repo_dropdown.value
                else None
            )
            descs = (
                [
                    _optimazation_id(mod)
                    for mod in self._all_models
                    if mod.optim_name != "base"
                ]
                if self._recal_checkbox.value
                else None
            )

            if self.filtered_callback:
                self.filtered_callback(datasets, repos, descs)

        def _recal_change(change):
            _invoke_callback()

        def _repos_change(change):
            self._selected_domain = change["new"]
            _invoke_callback()

        def _datasets_change(change):
            self._selected_dataset = change["new"]
            _invoke_callback()

        self._recal_checkbox.observe(_recal_change, names="value")
        self._repo_dropdown.observe(_repos_change, names="value")
        self._datasets_dropdown.observe(_datasets_change, names="value")


class ModelSelectWidgetContainer(object):
    """
    Widget used in model download notebooks for selecting a model for download

    :param filter_frameworks: if provided, will force all models
        to be one of these frameworks
    :param filter_datasets: if provided, will force all models
        to be trained on one of these datasets
    """

    def __init__(self, models: List[Model]):
        self._models = models
        self._models_widget = _ModelsWidget(self._models)
        self._filter_widget = _FilterWidget(self._models)

    @property
    def selected_model(self) -> Model:
        """
        :return: the selected model in the widget
        """
        return self._models_widget.selected

    @property
    def selected_framework(self) -> str:
        """
        :return: the selected framework in the widget
        """
        return self._models_widget.selected_framework

    def create(self):
        """
        :return: a created ipywidget that allows selection of models
        """

        def _filter_change_callback(
            datasets: List[str],
            repos: List[str],
            opt_names: List[str],
        ):
            filtered = []
            filtered = [
                mod
                for mod in self._models
                if (
                    (datasets is None or mod.dataset in datasets)
                    and (repos is None or mod.repo in repos)
                    and (opt_names is None or _optimazation_id(mod) in opt_names)
                )
            ]
            self._models_widget.update(filtered)

        self._filter_widget.filtered_callback = _filter_change_callback
        self._models_widget.update(self._models)

        return widgets.VBox(
            (
                widgets.HTML(format_html("Model Selector", header="h2")),
                self._filter_widget.container,
                self._models_widget.container,
            )
        )
