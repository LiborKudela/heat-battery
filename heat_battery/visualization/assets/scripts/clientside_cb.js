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

function GraphContainer(index, figure, updater_data) {
    var type = updater_data.updater_type;
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
    
    // Store transform_index in a variable for the click handler
    var transform_index = updater_data && updater_data.transform_index !== undefined ? updater_data.transform_index : null;
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
                            var editor_store_data = {id: graph_id_obj, type: type};
                            // If this is a transform graph, include transform_index
                            if (type === 'transform' && transform_index !== null && transform_index !== undefined) {
                                editor_store_data.transform_index = transform_index;
                            }
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
            l: 30,
            r: 10,
            t: 30,
            b: 50
        },
        xaxis: {},
        yaxis: {},
        title: {
            text: title_text,
            x: 0.5,
            y: 0.98,
            xanchor: 'center',
            yanchor: 'top',
            font: {
                size: 12,
            }
        }   
        }
    };
}

function add_graph_to_grid(n_clicks, title_text, currentChildren, updaters_data, plotly_type, updater_data) {
    // If no clicks, keep the current state
    if (!n_clicks) {
        return currentChildren;
    }

    // Initialize children array if empty
    if (!currentChildren) {
        currentChildren = [];
    }
    figure = get_empty_figure(title_text, plotly_type)  
    var graphIndex = currentChildren.length;
    var newGraph = GraphContainer(graphIndex, figure, updater_data);

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

// Deep merge function for nested objects (specifically for Plotly layout)
function deepMergeLayout(target, source) {
    for (const key in source) {
        if (source.hasOwnProperty(key)) {
            const sourceVal = source[key];
            const targetVal = target[key];
            
            // Check if both are non-null objects (and not arrays)
            if (sourceVal && typeof sourceVal === 'object' && !Array.isArray(sourceVal) && 
                targetVal && typeof targetVal === 'object' && !Array.isArray(targetVal)) {
                // Recursively merge nested objects (like yaxis, xaxis, etc.)
                deepMergeLayout(targetVal, sourceVal);
            } else {
                // For primitives, arrays, or if target doesn't have the key, assign directly
                target[key] = sourceVal;
            }
        }
    }
    return target;
}

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
        
        // Add any custom layout properties from the config (deep merge for nested objects like yaxis)
        if (figConfig.layout) {
            deepMergeLayout(figure.layout, figConfig.layout);
        }
        
        // Convert string titles to objects with explicit styling to ensure visibility
        if (figure.layout.xaxis && typeof figure.layout.xaxis.title === 'string') {
            figure.layout.xaxis.title = {
                text: figure.layout.xaxis.title,
                //font: { size: 14, color: '#ffffff' },
                standoff: 5
            };
        }
        if (figure.layout.yaxis && typeof figure.layout.yaxis.title === 'string') {
            figure.layout.yaxis.title = {
                text: figure.layout.yaxis.title,
                //font: { size: 14, color: '#ffffff' },
                standoff: 5
            };
        }

        if (figConfig.data) {
            // Replace figure data with config data, but preserve type if missing
            const originalType = figure.data[0]?.type || 'scattergl';
            figure.data = figConfig.data.map(trace => ({
                type: originalType,
                mode: 'lines',
                ...trace
            }));
        }
        var updater_data = {
            updater_type: figConfig.updater_type || 'timeseries',
            y_names: figConfig.y_names || null,
            x_name: figConfig.x_name || 't_timestamp'
        };
        // Include transform_index if it's a transform graph
        if (figConfig.transform_index !== undefined) {
            updater_data.transform_index = figConfig.transform_index;
            updater_data.transform_name = figConfig.transform_name;
        }
        const graphContainer = GraphContainer(index, figure, updater_data);
        gridChildren.push(graphContainer);
        
        // Add updater data
        updaters.push(updater_data);
    });
    
    return [gridChildren, updaters];
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        get_client_info: function(n_intervals) {
            var os = 'Unknown';
            if (navigator.userAgentData && navigator.userAgentData.platform) {
                os = navigator.userAgentData.getHighEntropyValues(["platform"])[0].value;
            } else if (navigator.platform) {
                os = navigator.platform;
            }
            var timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'Unknown';
            var language = navigator.language || navigator.userLanguage || 'Unknown';
            
            return {
                'os': os,
                'timezone': timezone,
                'language': language
            };
        },
        show_select_columns_modal: function(n_clicks) {
            return true;
        },
        open_data_transforms_modal: function(n_clicks) {
            console.log("Opening data transforms modal");
            return true;
        },
        close_data_transforms_modal: function(n_clicks) {
            console.log("Closing data transforms modal");
            return false;
        },
        // Initialize default transforms on page load
        initialize_default_transforms: function(initial_transforms) {
            if (!initial_transforms || initial_transforms.length === 0) {
                return window.dash_clientside.no_update;
            }
            try {
                var stored = localStorage.getItem('data_transforms');
                var existingTransforms = [];
                if (stored) {
                    existingTransforms = JSON.parse(stored);
                }
                
                // Convert old format transforms to new steps format
                existingTransforms = existingTransforms.map(function(transform) {
                    if (transform.steps || !transform.config) {
                        return transform; // Already in new format or no config
                    }
                    // Convert old config to steps
                    var config = transform.config;
                    var stepType = transform.type || 'column_transform';
                    var operation = config.operation || 'unknown';
                    var description = transform.description || operation;
                    
                    var step = {
                        type: stepType,
                        description: description,
                        operation: operation
                    };
                    
                    // Copy config properties to step
                    for (var key in config) {
                        if (config.hasOwnProperty(key) && key !== 'operation') {
                            step[key] = config[key];
                        }
                    }
                    
                    // Create new transform with steps
                    var newTransform = {};
                    for (var prop in transform) {
                        if (transform.hasOwnProperty(prop) && prop !== 'config' && prop !== 'type') {
                            newTransform[prop] = transform[prop];
                        }
                    }
                    newTransform.steps = [step];
                    return newTransform;
                });
                
                // Check if we need to merge defaults
                // Only add defaults that don't already exist (by id or name)
                var existingIds = new Set();
                var existingNames = new Set();
                existingTransforms.forEach(function(t) {
                    if (t.id) existingIds.add(t.id);
                    if (t.name) existingNames.add(t.name);
                });
                
                var newDefaults = [];
                initial_transforms.forEach(function(defaultTransform) {
                    // Check if this default already exists
                    var exists = false;
                    if (defaultTransform.id && existingIds.has(defaultTransform.id)) {
                        exists = true;
                    } else if (defaultTransform.name && existingNames.has(defaultTransform.name)) {
                        exists = true;
                    }
                    
                    if (!exists) {
                        // Ensure default transforms have an id
                        if (!defaultTransform.id) {
                            defaultTransform.id = 'default_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                        }
                        // Ensure defaults have steps array (convert if needed)
                        if (!defaultTransform.steps && defaultTransform.config) {
                            var config = defaultTransform.config;
                            var stepType = defaultTransform.type || 'column_transform';
                            var operation = config.operation || 'unknown';
                            var step = {
                                type: stepType,
                                description: defaultTransform.description || operation,
                                operation: operation
                            };
                            for (var key in config) {
                                if (config.hasOwnProperty(key) && key !== 'operation') {
                                    step[key] = config[key];
                                }
                            }
                            defaultTransform.steps = [step];
                            delete defaultTransform.config;
                            delete defaultTransform.type;
                        }
                        newDefaults.push(defaultTransform);
                        if (defaultTransform.id) existingIds.add(defaultTransform.id);
                        if (defaultTransform.name) existingNames.add(defaultTransform.name);
                    }
                });
                
                // Merge: existing first, then new defaults
                if (newDefaults.length > 0 || existingTransforms.length !== (stored ? JSON.parse(stored).length : 0)) {
                    var merged = existingTransforms.concat(newDefaults);
                    localStorage.setItem('data_transforms', JSON.stringify(merged));
                    return merged;
                }
                
                // If no new defaults, return existing
                if (existingTransforms.length > 0) {
                    return existingTransforms;
                }
                
                // If nothing exists, return initial transforms
                return initial_transforms;
            } catch (e) {
                console.error('Error initializing default transforms:', e);
                return initial_transforms || [];
            }
        },
        // Load transforms from localStorage when modal opens
        load_transforms_from_storage: function(is_open) {
            if (!is_open) {
                return window.dash_clientside.no_update;
            }
            try {
                var stored = localStorage.getItem('data_transforms');
                if (stored) {
                    return JSON.parse(stored);
                }
            } catch (e) {
                console.error('Error loading transforms from localStorage:', e);
            }
            return [];
        },
        // Display transforms list with expandable step editing
        display_transforms_list: function(transforms, expanded_index) {
            if (!transforms || transforms.length === 0) {
                return {
                    type: 'P',
                    namespace: 'dash_html_components',
                    props: {
                        children: 'No transforms defined yet.',
                        style: {color: '#888', fontStyle: 'italic', padding: '10px'}
                    }
                };
            }
            var isExpanded = expanded_index !== null && expanded_index !== undefined;
            var listItems = transforms.map(function(transform, index) {
                // Handle backward compatibility: convert old config format to steps
                var steps = transform.steps || [];
                if (steps.length === 0 && transform.config) {
                    var config = transform.config;
                    var stepType = transform.type || 'column_transform';
                    var operation = config.operation || 'unknown';
                    var description = transform.description || operation;
                    var step = {
                        type: stepType,
                        description: description,
                        operation: operation
                    };
                    for (var key in config) {
                        if (config.hasOwnProperty(key) && key !== 'operation') {
                            step[key] = config[key];
                        }
                    }
                    steps = [step];
                }
                
                var stepCount = steps.length;
                var stepText = stepCount === 0 ? 'No steps' : (stepCount === 1 ? '1 step' : stepCount + ' steps');
                var isThisExpanded = isExpanded && expanded_index === index;
                
                // Build collapsed view content
                var collapsedContent = [
                    {
                        type: 'Strong',
                        namespace: 'dash_html_components',
                        props: {children: transform.name || 'Unnamed Transform'}
                    },
                    {
                        type: 'Br',
                        namespace: 'dash_html_components',
                        props: {}
                    },
                    {
                        type: 'Small',
                        namespace: 'dash_html_components',
                        props: {
                            children: transform.description || 'No description',
                            style: {color: '#666'}
                        }
                    },
                    {
                        type: 'Br',
                        namespace: 'dash_html_components',
                        props: {}
                    },
                    {
                        type: 'Small',
                        namespace: 'dash_html_components',
                        props: {
                            children: stepText,
                            style: {color: '#888', fontWeight: 'bold', fontSize: '0.9em'}
                        }
                    }
                ];
                
                // Build expanded view with steps
                var expandedContent = [];
                if (isThisExpanded) {
                    expandedContent = [
                        {
                            type: 'Hr',
                            namespace: 'dash_html_components',
                            props: {style: {margin: '10px 0'}}
                        },
                        {
                            type: 'Strong',
                            namespace: 'dash_html_components',
                            props: {
                                children: 'Steps:',
                                style: {display: 'block', marginBottom: '10px'}
                            }
                        }
                    ];
                    
                    // Add each step with its config and controls
                    steps.forEach(function(step, stepIdx) {
                        var stepConfigItems = [];
                        // Show all step properties as config using Bootstrap styling
                        var configRows = [];
                        for (var key in step) {
                            if (step.hasOwnProperty(key) && key !== 'id') {
                                var value = step[key];
                                if (typeof value === 'object') {
                                    value = JSON.stringify(value);
                                }
                                configRows.push({
                                    type: 'Div',
                                    namespace: 'dash_html_components',
                                    props: {
                                        children: [
                                            {
                                                type: 'Strong',
                                                namespace: 'dash_html_components',
                                                props: {children: key + ': '}
                                            },
                                            {
                                                type: 'Code',
                                                namespace: 'dash_html_components',
                                                props: {
                                                    children: String(value),
                                                    //className: 'text-break'
                                                }
                                            }
                                        ],
                                        className: 'mb-2 p-2 border rounded'
                                    }
                                });
                            }
                        }
                        stepConfigItems = configRows;
                        
                        // Buttons div for step controls
                        var stepButtonsDiv = {
                                        type: 'Div',
                                        namespace: 'dash_html_components',
                                        props: {
                                            children: [
                                                {
                                                    type: 'Button',
                                                    namespace: 'dash_bootstrap_components',
                                                    props: {
                                                        children: '↑',
                                                        id: {'type': 'move-step-up-button', 'index': stepIdx},
                                                        color: 'secondary',
                                                        size: 'sm',
                                                        className: 'me-1',
                                                        disabled: stepIdx === 0,
                                                        n_clicks: 0
                                                    }
                                                },
                                                {
                                                    type: 'Button',
                                                    namespace: 'dash_bootstrap_components',
                                                    props: {
                                                        children: '↓',
                                                        id: {'type': 'move-step-down-button', 'index': stepIdx},
                                                        color: 'secondary',
                                                        size: 'sm',
                                                        className: 'me-1',
                                                        disabled: stepIdx === steps.length - 1,
                                                        n_clicks: 0
                                                    }
                                                },
                                                {
                                                    type: 'Button',
                                                    namespace: 'dash_bootstrap_components',
                                                    props: {
                                                        children: 'Preview',
                                                        id: {'type': 'preview-step-button', 'index': stepIdx},
                                                        color: 'info',
                                                        size: 'sm',
                                                        className: 'me-1',
                                                        n_clicks: 0
                                                    }
                                                },
                                                {
                                                    type: 'Button',
                                                    namespace: 'dash_bootstrap_components',
                                                    props: {
                                                        children: 'Remove',
                                                        id: {'type': 'remove-step-button', 'index': stepIdx},
                                                        color: 'danger',
                                                        size: 'sm',
                                                        n_clicks: 0
                                                    }
                                                }
                                            ],
                                            style: {
                                                display: 'flex',
                                                gap: '4px',
                                    alignItems: 'center',
                                    marginBottom: '10px'
                                }
                            }
                        };
                        
                        expandedContent.push({
                            type: 'Div',
                            namespace: 'dash_html_components',
                            props: {
                                children: [
                                    stepButtonsDiv,
                                    {
                                        type: 'Div',
                                        namespace: 'dash_html_components',
                                        props: {
                                            children: [
                                                {
                                                    type: 'H6',
                                                    namespace: 'dash_html_components',
                                                    props: {
                                                        children: 'Step ' + (stepIdx + 1) + ': ' + (step.description || step.type || 'Unknown'),
                                                        className: 'mb-3'
                                                    }
                                                }
                                            ].concat(stepConfigItems),
                                            className: 'flex-grow-1'
                                        }
                                    }
                                ],
                                style: {
                                    display: 'flex',
                                    flexDirection: 'column',
                                    marginBottom: '10px'
                                }
                            }
                        });
                    });
                    
                    // Add step button
                    expandedContent.push({
                        type: 'Button',
                        namespace: 'dash_bootstrap_components',
                        props: {
                            children: '+ Add Step',
                            id: {'type': 'add-step-button', 'index': index},
                            color: 'success',
                            size: 'sm',
                            n_clicks: 0
                        }
                    });
                }
                
                // Buttons div for collapse, preview, and delete
                var buttonsDiv = {
                                type: 'Div',
                                namespace: 'dash_html_components',
                                props: {
                                    children: [
                                        {
                                            type: 'Button',
                                            namespace: 'dash_bootstrap_components',
                                            props: {
                                    children: isThisExpanded ? 'Collapse' : 'Expand',
                                                id: {'type': 'edit-transform-button', 'index': index},
                                                color: 'primary',
                                                size: 'sm',
                                                className: 'me-2',
                                                n_clicks: 0
                                            }
                                        },
                            {
                                type: 'Button',
                                namespace: 'dash_bootstrap_components',
                                props: {
                                    children: 'Preview',
                                    id: {'type': 'preview-transform-button', 'index': index},
                                    color: 'info',
                                    size: 'sm',
                                    className: 'me-2',
                                    n_clicks: 0
                                }
                            },
                                        {
                                            type: 'Button',
                                            namespace: 'dash_bootstrap_components',
                                            props: {
                                                children: 'Delete',
                                                id: {'type': 'delete-transform-button', 'index': index},
                                                color: 'danger',
                                                size: 'sm',
                                                n_clicks: 0
                                            }
                                        }
                                    ],
                                    style: {
                                        display: 'flex',
                                        gap: '4px',
                            alignItems: 'flex-start',
                            marginBottom: '10px'
                        }
                    }
                };
                
                return {
                    type: 'ListGroupItem',
                    namespace: 'dash_bootstrap_components',
                    props: {
                        children: [
                            {
                                type: 'Div',
                                namespace: 'dash_html_components',
                                props: {
                                    children: [buttonsDiv].concat(collapsedContent).concat(expandedContent),
                                    style: {flex: 1, width: '100%'}
                                }
                            }
                        ],
                        action: false,
                        id: {'type': 'transform-item', 'index': index},
                        style: {
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'flex-start',
                            padding: '10px',
                            flexDirection: 'column',
                        }
                    }
                };
            });
            return listItems;
        },
        // Add transform
        add_transform: function(n_clicks, current_transforms) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            var transforms = current_transforms || [];
            var newTransform = {
                id: Date.now(),
                name: 'New Transform ' + (transforms.length + 1),
                description: 'Transform description',
                steps: []  // Start with empty steps array
            };
            transforms.push(newTransform);
            try {
                localStorage.setItem('data_transforms', JSON.stringify(transforms));
            } catch (e) {
                console.error('Error saving transforms to localStorage:', e);
            }
            return transforms;
        },
        // Toggle transform expansion
        toggle_transform_expansion: function(n_clicks_array, current_expanded) {
            if (!n_clicks_array) {
                return window.dash_clientside.no_update;
            }
            // Find which button was clicked
            var clickedIndex = -1;
            for (var i = 0; i < n_clicks_array.length; i++) {
                if (n_clicks_array[i] && n_clicks_array[i] > 0) {
                    clickedIndex = i;
                    break;
                }
            }
            if (clickedIndex < 0) {
                return window.dash_clientside.no_update;
            }
            // Toggle: if same index, collapse; otherwise expand new one
            if (current_expanded === clickedIndex) {
                return null; // Collapse
            } else {
                return clickedIndex; // Expand
            }
        },
        // Add step to transform
        add_step_to_transform: function(n_clicks_array, transforms, expanded_index) {
            if (!n_clicks_array || expanded_index === null || expanded_index === undefined) {
                return window.dash_clientside.no_update;
            }
            var clickedIndex = -1;
            for (var i = 0; i < n_clicks_array.length; i++) {
                if (n_clicks_array[i] && n_clicks_array[i] > 0) {
                    clickedIndex = i;
                    break;
                }
            }
            if (clickedIndex !== expanded_index || !transforms || expanded_index >= transforms.length) {
                return window.dash_clientside.no_update;
            }
            var transform = transforms[expanded_index];
            if (!transform.steps) {
                transform.steps = [];
            }
            var newStep = {
                type: 'column_selection',
                description: 'New step',
                operation: 'select',
                column_pattern: '.*'
            };
            transform.steps.push(newStep);
            try {
                localStorage.setItem('data_transforms', JSON.stringify(transforms));
            } catch (e) {
                console.error('Error saving transforms to localStorage:', e);
            }
            return transforms;
        },
        // Remove step from transform
        remove_step_from_transform: function(n_clicks_array, transforms, expanded_index) {
            if (!n_clicks_array || expanded_index === null || expanded_index === undefined) {
                return window.dash_clientside.no_update;
            }
            var clickedStepIndex = -1;
            for (var i = 0; i < n_clicks_array.length; i++) {
                if (n_clicks_array[i] && n_clicks_array[i] > 0) {
                    clickedStepIndex = i;
                    break;
                }
            }
            if (clickedStepIndex < 0 || !transforms || expanded_index >= transforms.length) {
                return window.dash_clientside.no_update;
            }
            var transform = transforms[expanded_index];
            if (!transform.steps || clickedStepIndex >= transform.steps.length) {
                return window.dash_clientside.no_update;
            }
            transform.steps.splice(clickedStepIndex, 1);
            try {
                localStorage.setItem('data_transforms', JSON.stringify(transforms));
            } catch (e) {
                console.error('Error saving transforms to localStorage:', e);
            }
            return transforms;
        },
        // Move step up
        move_step_up: function(n_clicks_array, transforms, expanded_index) {
            if (!n_clicks_array || expanded_index === null || expanded_index === undefined) {
                return window.dash_clientside.no_update;
            }
            var clickedStepIndex = -1;
            for (var i = 0; i < n_clicks_array.length; i++) {
                if (n_clicks_array[i] && n_clicks_array[i] > 0) {
                    clickedStepIndex = i;
                    break;
                }
            }
            if (clickedStepIndex <= 0 || !transforms || expanded_index >= transforms.length) {
                return window.dash_clientside.no_update;
            }
            var transform = transforms[expanded_index];
            if (!transform.steps || clickedStepIndex >= transform.steps.length) {
                return window.dash_clientside.no_update;
            }
            // Swap with previous step
            var temp = transform.steps[clickedStepIndex];
            transform.steps[clickedStepIndex] = transform.steps[clickedStepIndex - 1];
            transform.steps[clickedStepIndex - 1] = temp;
            try {
                localStorage.setItem('data_transforms', JSON.stringify(transforms));
            } catch (e) {
                console.error('Error saving transforms to localStorage:', e);
            }
            return transforms;
        },
        // Move step down
        move_step_down: function(n_clicks_array, transforms, expanded_index) {
            if (!n_clicks_array || expanded_index === null || expanded_index === undefined) {
                return window.dash_clientside.no_update;
            }
            var clickedStepIndex = -1;
            for (var i = 0; i < n_clicks_array.length; i++) {
                if (n_clicks_array[i] && n_clicks_array[i] > 0) {
                    clickedStepIndex = i;
                    break;
                }
            }
            if (clickedStepIndex < 0 || !transforms || expanded_index >= transforms.length) {
                return window.dash_clientside.no_update;
            }
            var transform = transforms[expanded_index];
            if (!transform.steps || clickedStepIndex >= transform.steps.length - 1) {
                return window.dash_clientside.no_update;
            }
            // Swap with next step
            var temp = transform.steps[clickedStepIndex];
            transform.steps[clickedStepIndex] = transform.steps[clickedStepIndex + 1];
            transform.steps[clickedStepIndex + 1] = temp;
            try {
                localStorage.setItem('data_transforms', JSON.stringify(transforms));
            } catch (e) {
                console.error('Error saving transforms to localStorage:', e);
            }
            return transforms;
        },
        // Close step preview modal
        close_step_preview_modal: function(n_clicks) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            return false;
        },
        // Delete transform
        delete_transform: function(n_clicks_array, current_transforms) {
            if (!n_clicks_array || !current_transforms) {
                return window.dash_clientside.no_update;
            }
            // Find which button was clicked
            var clickedIndex = -1;
            for (var i = 0; i < n_clicks_array.length; i++) {
                if (n_clicks_array[i] && n_clicks_array[i] > 0) {
                    clickedIndex = i;
                    break;
                }
            }
            if (clickedIndex < 0 || clickedIndex >= current_transforms.length) {
                return window.dash_clientside.no_update;
            }
            var transforms = current_transforms || [];
            if (transforms.length === 0) {
                return window.dash_clientside.no_update;
            }
            var transformToDelete = transforms[clickedIndex];
            var transformName = transformToDelete.name || 'Unnamed Transform';
            if (confirm('Delete transform "' + transformName + '"?')) {
                transforms = transforms.filter(function(t, idx) {
                    return idx !== clickedIndex;
                });
                try {
                    localStorage.setItem('data_transforms', JSON.stringify(transforms));
                } catch (e) {
                    console.error('Error saving transforms to localStorage:', e);
                }
                return transforms;
            }
            return window.dash_clientside.no_update;
        },
        // Line click modal callbacks
        open_line_click_modal: function(clickData, variableDescriptions) {
            if (!clickData) {
                return [false, window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            
            var point = clickData.point;
            var graphId = clickData.graphId;
            
            // Check if this is a "no line selected" case
            if (clickData.message) {
                return [true, 'No Line Selected', clickData.message];
            }
            
            var title = `Line: ${point.traceName || 'Unknown'}`;
            
            // Look up description for this variable
            var traceName = point.traceName || '';
            var description = 'Missing description';
            if (variableDescriptions && variableDescriptions[traceName]) {
                description = variableDescriptions[traceName];
            }
            
            // Build modal body content
            var bodyContent = {
                type: "Div",
                namespace: "dash_html_components",
                props: {
                    children: [
                        {
                            type: "P",
                            namespace: "dash_html_components",
                            props: {
                                children: `Trace Name: ${point.traceName || 'N/A'}`,
                                style: {fontWeight: 'bold', marginBottom: '8px'}
                            }
                        },
                        {
                            type: "Div",
                            namespace: "dash_html_components",
                            props: {
                                children: [
                                    {
                                        type: "Strong",
                                        namespace: "dash_html_components",
                                        props: {children: 'Description: '}
                                    },
                                    {
                                        type: "Span",
                                        namespace: "dash_html_components",
                                        props: {
                                            children: description,
                                            style: {fontStyle: description === 'Missing description' ? 'italic' : 'normal', color: description === 'Missing description' ? 'gray' : 'inherit'}
                                        }
                                    }
                                ],
                                style: {marginBottom: '12px', padding: '8px', backgroundColor: 'rgba(128,128,128,0.1)', borderRadius: '4px'}
                            }
                        },
                        {
                            type: "Hr",
                            namespace: "dash_html_components",
                            props: {}
                        },
                        {
                            type: "P",
                            namespace: "dash_html_components",
                            props: {
                                children: `X Value: ${point.x !== undefined ? point.x : 'N/A'}`
                            }
                        },
                        {
                            type: "P",
                            namespace: "dash_html_components",
                            props: {
                                children: `Y Value: ${point.y !== undefined ? point.y : 'N/A'}`
                            }
                        },
                        {
                            type: "P",
                            namespace: "dash_html_components",
                            props: {
                                children: `Point Index: ${point.pointIndex !== undefined ? point.pointIndex : 'N/A'}`
                            }
                        },
                        {
                            type: "P",
                            namespace: "dash_html_components",
                            props: {
                                children: `Curve Number: ${point.curveNumber !== undefined ? point.curveNumber : 'N/A'}`
                            }
                        },
                        {
                            type: "Hr",
                            namespace: "dash_html_components",
                            props: {}
                        },
                        {
                            type: "P",
                            namespace: "dash_html_components",
                            props: {
                                children: `Graph ID: ${JSON.stringify(graphId)}`,
                                style: {fontSize: '0.85em', color: 'gray'}
                            }
                        }
                    ]
                }
            };
            
            return [true, title, bodyContent];
        },
        
        close_line_click_modal: function(n_clicks) {
            return false;
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
                x_name: 't_timestamp',
            }
            return add_graph_to_grid(n_clicks, 'Timeseries data', currentChildren, updaters_data, 'scattergl', new_updater_data);    
        },
        // Update transform graphs menu items
        update_transform_graphs_menu: function(transforms) {
            if (!transforms || transforms.length === 0) {
                return [];
            }
            var menuItems = [];
            for (var i = 0; i < transforms.length; i++) {
                var transform = transforms[i];
                var transformName = transform.name || ('Transform ' + (i + 1));
                menuItems.push({
                    type: 'DropdownMenuItem',
                    namespace: 'dash_bootstrap_components',
                    props: {
                        children: transformName,
                        id: {'type': 'add-transform-graph-button', 'index': i},
                        n_clicks: 0
                    }
                });
            }
            return menuItems;
        },
        // Add transform graph to grid
        add_transform_graph_to_grid: function(n_clicks_array, transforms, currentChildren, updaters_data) {
            if (!n_clicks_array || !transforms) {
                return window.dash_clientside.no_update;
            }
            
            // Find which transform button was clicked
            var clickedIndex = -1;
            for (var i = 0; i < n_clicks_array.length; i++) {
                if (n_clicks_array[i] && n_clicks_array[i] > 0) {
                    clickedIndex = i;
                    break;
                }
            }
            
            if (clickedIndex < 0 || clickedIndex >= transforms.length) {
                return window.dash_clientside.no_update;
            }
            
            var transform = transforms[clickedIndex];
            var transformName = transform.name || ('Transform ' + (clickedIndex + 1));
            
            var new_updater_data = {
                updater_type: 'transform',
                transform_index: clickedIndex,
                transform_name: transformName,
                y_names: null,
                x_name: 't_timestamp',
            };
            
            return add_graph_to_grid(n_clicks_array[clickedIndex], 'Transform data (' + transformName + ')', currentChildren, updaters_data, 'scattergl', new_updater_data);
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
            
            // Preserve existing updater data, especially for transform graphs
            var existing_updater = updaters_data[index] || {};
            var chart_type = chart_editor_store_data.type;
            
            if (chart_type === 'transform' && existing_updater.transform_index !== undefined) {
                // Preserve transform_index and transform_name for transform graphs
                updaters_data[index] = {
                    updater_type: 'transform',
                    transform_index: existing_updater.transform_index,
                    transform_name: existing_updater.transform_name,
                    y_names: y_names,
                    x_name: existing_updater.x_name || 't_timestamp',
                };
            } else {
                updaters_data[index] = {
                    updater_type: 'timeseries',
                    y_names: y_names,
                    x_name: 't_timestamp',
                };
            }
            console.log("Updating figure and updaters data");
            dash_clientside.set_props('figure-updaters-store', {data: updaters_data});
            // Use Plotly.react to replace the entire figure (data + layout)
            Plotly.react(plotly_obj, figure.data, figure.layout);
        },
        initialize_figures: function(initial_figures, signature) {
            return initialize_figures_on_load(initial_figures, signature);
        },
        
        // ========================================================================
        // COMPARATOR CALLBACKS
        // ========================================================================
        
        add_comparator_scatter_chart: function(n_clicks, currentChildren, updaters_data) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            
            var new_updater_data = {
                chart_type: 'scatter',
                x_column: null,
                y_column: null,
            };
            
            return add_comparator_chart_to_grid(n_clicks, currentChildren, updaters_data, 'scatter', new_updater_data);
        },
        
        add_comparator_bar_chart: function(n_clicks, currentChildren, updaters_data) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            
            var new_updater_data = {
                chart_type: 'bar',
                x_column: null,
                y_column: null,
            };
            
            return add_comparator_chart_to_grid(n_clicks, currentChildren, updaters_data, 'bar', new_updater_data);
        },
        
        add_comparator_box_chart: function(n_clicks, currentChildren, updaters_data) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            
            var new_updater_data = {
                chart_type: 'box',
                x_column: null,
                y_column: null,
            };
            
            return add_comparator_chart_to_grid(n_clicks, currentChildren, updaters_data, 'box', new_updater_data);
        },
        
        remove_comparator_chart: function(remove_index, grid_div_children, updaters_data) {
            if (!remove_index || remove_index.length === 0) {
                return window.dash_clientside.no_update;
            }
            
            var index = remove_index[0];
            grid_div_children.splice(index, 1);
            updaters_data.splice(index, 1);
            
            // Re-index remaining charts
            for (let i = 0; i < grid_div_children.length; i++) {
                if (grid_div_children[i].props && grid_div_children[i].props.id) {
                    grid_div_children[i].props.id.index = i;
                }
            }
            
            return [grid_div_children, [], updaters_data];
        },
        
        save_and_close_comparator_chart_editor: function(n_clicks) {
            console.log("Saving comparator chart and closing editor");
            return [false, true];
        },
        
        save_comparator_chart_editor: function(n_clicks) {
            console.log("Saving comparator chart");
            return true;
        },
        
        close_no_save_comparator_chart_editor: function(n_clicks) {
            console.log("Closing comparator chart editor without saving");
            return false;
        },
        
        replace_comparator_figure_on_save: function(saveState, figure, chart_editor_store_data, updaters_data) {
            // Check for initial editor load (same pattern as result_viewer)
            if (figure.data && figure.data.length === 0) {
                console.log("Initial load of comparator chart editor");
                return;
            }
            
            var plotly_obj = document.getElementById(`${stringify_sorted(chart_editor_store_data.id)}`).getElementsByClassName("js-plotly-plot")[0];
            var index = Number(chart_editor_store_data.id.index);
            
            // Extract x and y columns from figure
            var x_column = figure.data[0].xsrc || null;
            var y_column = figure.data[0].ysrc || null;
            
            updaters_data[index] = {
                chart_type: chart_editor_store_data.chart_type,
                x_column: x_column,
                y_column: y_column,
            };
            
            console.log("Updating comparator figure and updaters data");
            dash_clientside.set_props('comparator-updaters-store', {data: updaters_data});
            // Use same pattern as result_viewer - pass figure object, not just layout
            Plotly.react(plotly_obj, figure.data, figure.layout);
        },
        
        initialize_comparator_charts: function(initial_figures, dataset) {
            if (!initial_figures || initial_figures.length === 0) {
                let status = dataset && dataset.length > 0 
                    ? `Dataset loaded: ${dataset.length} jobs - add charts using the dropdown`
                    : 'No initial charts configured - click "Refresh dataset" to load data';
                return [[], [], status];
            }
            
            let gridChildren = [];
            let updaters = [];
            
            // Create charts from initial configuration
            let chartIndex = 0;
            initial_figures.forEach((figConfig) => {
                if (!figConfig.active) return;
                
                const figure = get_empty_figure(
                    figConfig.title || 'Comparator Chart', 
                    figConfig.chart_type || 'scatter'
                );
                
                // Add custom layout properties
                if (figConfig.layout) {
                    deepMergeLayout(figure.layout, figConfig.layout);
                }
                
                // Convert string titles to objects with explicit styling to ensure visibility
                if (figure.layout.xaxis && typeof figure.layout.xaxis.title === 'string') {
                    figure.layout.xaxis.title = {
                        text: figure.layout.xaxis.title,
                        standoff: 5
                    };
                }
                if (figure.layout.yaxis && typeof figure.layout.yaxis.title === 'string') {
                    figure.layout.yaxis.title = {
                        text: figure.layout.yaxis.title,
                        standoff: 5
                    };
                }
                
                // If dataset is available and columns are specified, populate the chart
                if (dataset && dataset.length > 0 && figConfig.x_column && figConfig.y_column) {
                    const x_data = dataset.map(row => row[figConfig.x_column]);
                    const y_data = dataset.map(row => row[figConfig.y_column]);
                    
                    // Check if data exists
                    if (x_data.some(v => v !== undefined) && y_data.some(v => v !== undefined)) {
                        figure.data[0].x = x_data;
                        figure.data[0].y = y_data;
                        figure.data[0].mode = 'markers';
                        figure.data[0].name = figConfig.y_column;
                    }
                }
                
                // Create the graph container
                const graphContainer = ComparatorChartContainer(chartIndex, figure, figConfig.chart_type);
                gridChildren.push(graphContainer);
                
                // Add updater data
                updaters.push({
                    chart_type: figConfig.chart_type || 'scatter',
                    x_column: figConfig.x_column || null,
                    y_column: figConfig.y_column || null,
                });
                
                chartIndex++;
            });
            
            // Set status based on whether dataset is loaded
            let status;
            if (!dataset || dataset.length === 0) {
                status = `${initial_figures.filter(f => f.active).length} chart(s) ready - click "Refresh dataset" to load data`;
            } else {
                status = `Dataset loaded: ${dataset.length} jobs, ${initial_figures.filter(f => f.active).length} charts`;
            }
            
            return [gridChildren, updaters, status];
        }
    }
});

// ============================================================================
// COMPARATOR HELPER FUNCTIONS
// ============================================================================

function add_comparator_chart_to_grid(n_clicks, currentChildren, updaters_data, chart_type, updater_data) {
    if (!n_clicks) {
        return currentChildren;
    }
    
    if (!currentChildren) {
        currentChildren = [];
    }
    
    const figure = get_empty_figure(`${chart_type.charAt(0).toUpperCase() + chart_type.slice(1)} Chart`, chart_type);
    const newChart = ComparatorChartContainer(currentChildren.length, figure, chart_type);
    
    currentChildren.push(newChart);
    updaters_data.push(updater_data);
    
    return [currentChildren, updaters_data];
}

function ComparatorChartContainer(index, figure, chart_type) {
    var grid_item_id_obj = {type: 'comparator-grid-item', index: `container-${index}`};
    var fig_id_obj = {type: 'comparator-graph-container', index: index};
    var fig_id_str = stringify_sorted(fig_id_obj);
    var graph_id_obj = {type: 'comparator-graph', index: index.toString()};
    var graph_id_str = stringify_sorted(graph_id_obj);
    
    var graph = {
        type: "Graph",
        namespace: "dash_core_components",
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
                            }
                        }
                    },
                    {
                        name: "delete",
                        title: "Delete chart",
                        icon: myIcons.DeleteIcon,
                        click: function(gd) {
                            dash_clientside.set_props('remove-comparator-chart-store', {data: [index]});
                        }   
                    },
                    {
                        name: "edit",
                        title: "Edit chart",
                        icon: myIcons.EditIcon,
                        click: function(gd) {
                            var figure = document.getElementById(`${graph_id_str}`).getElementsByClassName("js-plotly-plot")[0];
                            var editor_store_data = {id: graph_id_obj, chart_type: chart_type};
                            dash_clientside.set_props('comparator-chart-editor-modal', {is_open: true});
                            dash_clientside.set_props('comparator-chart-editor-modal-title', {children: "Chart editor - " + chart_type});
                            dash_clientside.set_props('comparator-chart-editor-store', {data: editor_store_data});
                            dash_clientside.set_props('comparator-chart-editor-editor', {loadFigure: figure});
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

// ============================================================================
// GLOBAL RIGHT-CLICK HANDLER FOR PLOTLY GRAPHS
// ============================================================================

// Attach right-click (context menu) handler to all Plotly graphs
// Using capture phase to intercept before other handlers
document.addEventListener('contextmenu', function(event) {
    // Find the closest plotly graph element
    var plotlyGraph = event.target.closest('.js-plotly-plot');
    if (!plotlyGraph) return;
    
    // Prevent default context menu immediately when on a plotly graph
    event.preventDefault();
    event.stopPropagation();
    
    // Check if we have hover data (point under cursor)
    var hoverData = plotlyGraph._hoverdata;
    if (!hoverData || hoverData.length === 0) {
        // No point hovered - still show our modal but with a message
        dash_clientside.set_props('line-click-store', {data: {
            point: {
                traceName: 'No line selected',
                x: 'N/A',
                y: 'N/A',
                pointIndex: 'N/A',
                curveNumber: 'N/A'
            },
            graphId: null,
            timestamp: Date.now(),
            message: 'Hover over a line/point before right-clicking to see details'
        }});
        return;
    }
    
    // Get the first hovered point
    var point = hoverData[0];
    
    // Extract graph ID from the plotly element
    var graphContainer = plotlyGraph.closest('[id]');
    var graphId = null;
    if (graphContainer) {
        try {
            graphId = JSON.parse(graphContainer.id);
        } catch (e) {
            graphId = graphContainer.id;
        }
    }
    
    // Build click data object similar to Dash's clickData format
    var clickData = {
        point: {
            x: point.x,
            y: point.y,
            pointIndex: point.pointIndex,
            curveNumber: point.curveNumber,
            traceName: point.data ? point.data.name : (point.fullData ? point.fullData.name : 'Unknown'),
            customdata: point.customdata
        },
        graphId: graphId,
        timestamp: Date.now()
    };
    
    // Update the store to trigger the modal
    dash_clientside.set_props('line-click-store', {data: clickData});
}, true);  // Use capture phase

