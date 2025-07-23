function pad_number(n, n_digit_max) {
    const has_n_digits = n.toString().length;
    if (has_n_digits < n_digit_max) {
        return n.toString().padStart(n_digit_max-has_n_digits, ' ');
    } else {
        return n.toString();
    };
}

function obj2pydict(obj, comment_prefix="#") {
    return comment_prefix + "\n\n" + JSON.stringify(obj, null, 4).replace("true", "true").replace("false", "false");
}

function stringify_sorted (json_obj) {
    const all_keys = Object.keys(json_obj);
    const sorted_keys = all_keys.sort();
    const sorted_obj = {};
    sorted_keys.forEach((key) => {
        sorted_obj[key] = json_obj[key];
    });
    return JSON.stringify(sorted_obj);
}

myIcons = {
    LegendIcon: {
        width: 24,
        height: 24,
        path: 'M4 11a2 2 0 1 0-2-2 2 2 0 0 0 2 2zm0-3a1 1 0 1 1-1 1 1 1 0 0 1 1-1zm5-5h13v1H9zm0 6h13v1H9zm0 6h13v1H9zm0 6h13v1H9zm-7.063 2h4.126L4 19.209zm1.683-1l.38-.699.38.699zM6 1H2v4h4zM5 4H3V2h2zm1 9H2v4h4zm-1 3H3v-2h2z'
    },
    FullscreenIcon: {
        width: 500,
        height: 500,
        path: 'M200 32L56 32C42.7 32 32 42.7 32 56l0 144c0 9.7 5.8 18.5 14.8 22.2s19.3 1.7 26.2-5.2l40-40 79 79-79 79L73 295c-6.9-6.9-17.2-8.9-26.2-5.2S32 302.3 32 312l0 144c0 13.3 10.7 24 24 24l144 0c9.7 0 18.5-5.8 22.2-14.8s1.7-19.3-5.2-26.2l-40-40 79-79 79 79-40 40c-6.9 6.9-8.9 17.2-5.2 26.2s12.5 14.8 22.2 14.8l144 0c13.3 0 24-10.7 24-24l0-144c0-9.7-5.8-18.5-14.8-22.2s-19.3-1.7-26.2 5.2l-40 40-79-79 79-79 40 40c6.9 6.9 17.2 8.9 26.2 5.2s14.8-12.5 14.8-22.2l0-144c0-13.3-10.7-24-24-24L312 32c-9.7 0-18.5 5.8-22.2 14.8s-1.7 19.3 5.2 26.2l40 40-79 79-79-79 40-40c6.9-6.9 8.9-17.2 5.2-26.2S209.7 32 200 32z'
    },
    EditIcon: {
        width: 24,
        height: 24,
        path: 'M21,12a1,1,0,0,0-1,1v6a1,1,0,0,1-1,1H5a1,1,0,0,1-1-1V5A1,1,0,0,1,5,4h6a1,1,0,0,0,0-2H5A3,3,0,0,0,2,5V19a3,3,0,0,0,3,3H19a3,3,0,0,0,3-3V13A1,1,0,0,0,21,12ZM6,12.76V17a1,1,0,0,0,1,1h4.24a1,1,0,0,0,.71-.29l6.92-6.93h0L21.71,8a1,1,0,0,0,0-1.42L17.47,2.29a1,1,0,0,0-1.42,0L13.23,5.12h0L6.29,12.05A1,1,0,0,0,6,12.76ZM16.76,4.41l2.83,2.83L18.17,8.66,15.34,5.83ZM8,13.17l5.93-5.93,2.83,2.83L10.83,16H8Z'
    },
    DeleteIcon: {
        width: 24,
        height: 24,
        path: 'M7 4a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2h4a1 1 0 1 1 0 2h-1.069l-.867 12.142A2 2 0 0 1 17.069 22H6.93a2 2 0 0 1-1.995-1.858L4.07 8H3a1 1 0 0 1 0-2h4V4zm2 2h6V4H9v2zM6.074 8l.857 12H17.07l.857-12H6.074zM10 10a1 1 0 0 1 1 1v6a1 1 0 1 1-2 0v-6a1,1,0,0,1,1-1zm4 0a1 1 0 0 1 1 1v6a1 1 0 1 1-2 0v-6a1,1,0,0,1,1-1z'
    }
};

function GraphContainer(index, figure, type) {
    var grid_item_id_obj = {type: 'grid-item', index: `container-${index}`};
    var grid_item_id_str = stringify_sorted(grid_item_id_obj);
    var fig_id_obj = {type: 'graph-container', index: index};
    var fig_id_str = stringify_sorted(fig_id_obj);
    var relayout_id_obj = {type: 'relayout-store', index: index};
    var relayout_id_str = stringify_sorted(relayout_id_obj);
    // var updater_data_id_obj = {type: 'updater-data-store', index: index};
    // var updater_data_id_str = stringify_sorted(updater_data_id_obj);
    var graph_id_obj = {type: 'graph', index: index.toString()};
    var graph_id_str = stringify_sorted(graph_id_obj);
    // var relayout_store = {
    //     type: "Store",
    //     namespace: "dash_core_components",
    //     props: {
    //         id: relayout_id_obj,
    //         data: {
    //             autosize: true
    //         }
    //     }
    // };
    var graph = {
        type: "Graph",
        namespace: "dash_core_components",  // ensures it's a dcc.Graph
        props: {
            id: graph_id_obj,
            figure: figure,
            config: {
                responsive: true,
                scrollZoom: false,
                displaylogo: false,
                displayModeBar: "hover",
                modeBarButtonsToAdd: [
                    {
                        name: "toggleLegend",
                        title: "Toggle legend",
                        icon: myIcons.LegendIcon,
                        click: function(gd) {
                            var plotly_obj = document.getElementById(`${fig_id_str}`).getElementsByClassName("js-plotly-plot")[0];
                            var value = plotly_obj.layout.showlegend;
                            var update = {'showlegend': !value};
                            Plotly.update(plotly_obj, {}, update);
                            console.log('hello from custom modbar');
                        }
                    },
                    {
                        name: "toggleFullscreen",
                        title: "Toggle fullscreen",
                        icon: myIcons.FullscreenIcon,
                        click: function(gd) {
                            var target = document.getElementById(`${fig_id_str}`);
                            if (document.fullscreenElement === null) {
                                target.requestFullscreen();
                            } else {
                                document.exitFullscreen();
                            };
                            console.log('hello from custom modbar');
                        }
                    },
                    {
                        name: "delete",
                        title: "Delete graph",
                        icon: myIcons.DeleteIcon,
                        click: function(gd) {
                            //var fig_div = document.getElementById(`${fig_id_str}`);
                            dash_clientside.set_props('remove-figure-store', {data: [index]});
                            //fig_div.remove();
                        }   
                    },
                    {
                        name: "edit",
                        title: "Edit graph",
                        icon: myIcons.EditIcon,
                        click: function(gd) {
                            var figure = document.getElementById(`${graph_id_str}`).getElementsByClassName("js-plotly-plot")[0];
                            // dash_clientside.set_props('chart-editor-editor', {dataSources: dataSources});
                            var editor_store_data = {id: graph_id_obj, type: type};
                            dash_clientside.set_props('chart-editor-modal', {is_open: true}); //open the editor modal
                            dash_clientside.set_props('chart-editor-modal-title', {children: "Chart editor - " + type}); //set the title of the editor modal    
                            dash_clientside.set_props('chart-editor-store', {data: editor_store_data}); //set outp  ut id for the editor
                            //dash_clientside.set_props('chart-editor-editor', {saveState: false}); //set saveState to false
                            dash_clientside.set_props('chart-editor-editor', {loadFigure: figure}); //load the figure into the editor
                        }
                    }
                ]
            },
            style: {
                margin: "2px",
                borderRadius: "10px",
                overflow: "hidden",
                height: "100%"
            },
            animate: false
        }
    };
    var container = {
        type: "Div",
        namespace: "dash_html_components",
        props: {
            id: fig_id_obj,
            className: "grid-item",
            children: graph,
            style: {
                display: "flex",
                flexDirection: "column",
                height: "100%"
            }
        }
    };
    return container;
}

function get_empty_figure(title_text, plotly_type) {
    return {
        data: [{
            type: plotly_type,  
    }], 
    layout: {
        template: window.CustomPlotlyTemplates['plotly_dark'],
        showlegend: false,
        legend: {
            orientation: "v",
            yanchor: "top",
            y: 0.99,
            xanchor: "left",
            x: 0.01,
            bordercolor: "lightgray",
            borderwidth: 2,
        },
        margin: {
            l: 10,
            r: 10,
            t: 10,
            b: 10
        },
        title: {
            text: title_text,
            x: 0.5,
            y: 0.98,
            xanchor: 'center',
            yanchor: 'top',
            font: {
                size: 10,
            }
        }   
        }
    };
}

function add_graph_to_grid(n_clicks, currentChildren, updaters_data, plotly_type, updater_data) {
    // If no clicks, keep the current state
    if (!n_clicks) {
        return currentChildren;
    }

    // Initialize children array if empty
    if (!currentChildren) {
        currentChildren = [];
    }
    figure = get_empty_figure("Timeseries data", plotly_type)  
    var newGraph = GraphContainer(currentChildren.length, figure, updater_data.updater_type);

    // Append the new div to the current children array
    currentChildren.push(newGraph);

    // var updater_data = {
    //     updater_type: data_source_type,
    //     y_names: null,
    //     x_name: 't_sim_days',
    // }

    updaters_data.push(updater_data);
    return [currentChildren, updaters_data];  
};

//const py_icon = React.createElement(window.dash_iconify.DashIconify, {icon:"feather:info", width: 20});

function initialize_figures_on_load(initial_figures, signature) {
    // If no initial figures or no signature, return empty defaults
    if (!initial_figures || !initial_figures.length || !signature) {
        return [[], []];
    }
    
    let gridChildren = [];
    let updaters = [];
    
    // Create figures from initial configuration
    initial_figures.forEach((figConfig, index) => {
        const figure = get_empty_figure(figConfig.title || "Auto-generated figure", figConfig.plotly_type || 'scattergl');
        
        
        // Add any custom layout properties from the config
        if (figConfig.layout) {
            Object.assign(figure.layout, figConfig.layout);
        }
        if (figConfig.data) {
            Object.assign(figure.data, figConfig.data);
        // Create the graph container
        }
        const graphContainer = GraphContainer(index, figure, figConfig.updater_type);
        gridChildren.push(graphContainer);
        
        // Add updater data
        updaters.push({
            updater_type: figConfig.updater_type || 'timeseries',
            y_names: figConfig.y_names || null,
            x_name: figConfig.x_name || 't_timestamp'
        });
    });
    
    return [gridChildren, updaters];
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        show_select_columns_modal: function(n_clicks) {
            return true;
        },
        showhide_columns: function(value, columnDefs) {
            // Create a new array with the updated column definitions
            var newColumnDefs = [...columnDefs];    
            for (let i = 0; i < newColumnDefs.length; i++) {
                newColumnDefs[i].hide = !value.includes(i);
            };
            return newColumnDefs;
        },
        add_timeseries_graph_to_grid: function(n_clicks, currentChildren, updaters_data) {
            var new_updater_data = {
                updater_type: 'timeseries',
                y_names: null,
                x_name: 't_sim_days',
            }
            return add_graph_to_grid(n_clicks, currentChildren, updaters_data, 'scattergl', new_updater_data);    
        },
        add_last_value_aggregated_graph_to_grid: function(n_clicks, currentChildren, updaters_data) {
            var new_updater_data = {
                updater_type: 'last_value_aggregated',
                y_names: null,
                x_name: 't_sim_days',
            }
            return add_graph_to_grid(n_clicks, currentChildren, updaters_data, 'scattergl', new_updater_data);   
        },  
        add_monthly_aggregated_graph_to_grid: function(n_clicks, currentChildren, updaters_data) {
            var new_updater_data = {
                updater_type: 'monthly_aggregated',
                y_names: null,
                x_name: 't_sim_days',
            }
            return add_graph_to_grid(n_clicks, currentChildren, updaters_data, 'scattergl', new_updater_data);   
        },
        add_yearly_aggregated_graph_to_grid: function(n_clicks, currentChildren, updaters_data) {
            var new_updater_data = {
                updater_type: 'yearly_aggregated',
                y_names: null,
                x_name: 't_sim_days',
            }
            return add_graph_to_grid(n_clicks, currentChildren, updaters_data, 'scattergl', new_updater_data);   
        },
        remove_figure_from_grid: function(remove_index, grid_div_children, updaters_data) {
            grid_div_children.splice(remove_index, 1);
            updaters_data.splice(remove_index, 1);
            return [grid_div_children, [], updaters_data];
        },
        save_and_close_chart_editor: function(n_clicks) {
            console.log("Saving figure changes to grid and closing chart editor");
            return [false, true];
        },
        save_chart_editor: function(n_clicks) {   
            console.log("Figure changes saved to grid");
            return true; //set modal saveState to true
        },
        close_no_save_chart_editor: function(n_clicks) {
            console.log("Closing chart editor without saving changes to grid");
            return false; //set modal is_open to false
        },
        replace_figure_on_save: function(saveState, figure, chart_editor_store_data, updaters_data) {

            //check for initial editor load
            if (figure.data && figure.data.length === 0) {
                console.log("Initial load of chart editor");
                return;
            }

            var plotly_obj = document.getElementById(`${stringify_sorted(chart_editor_store_data.id)}`).getElementsByClassName("js-plotly-plot")[0];
            var index = Number(chart_editor_store_data.id.index);
            var y_names = figure.data.map((trace) => trace.ysrc);
            updaters_data[index] = {
                updater_type: 'timeseries',
                y_names: y_names,
                x_name: 't_timestamp',
            };
            console.log("Updating figure and updaters data");
            dash_clientside.set_props('figure-updaters-store', {data: updaters_data});
            Plotly.update(plotly_obj, {}, {figure: figure});
        },
        initialize_figures: function(initial_figures, signature) {
            return initialize_figures_on_load(initial_figures, signature);
        }
    }
});


