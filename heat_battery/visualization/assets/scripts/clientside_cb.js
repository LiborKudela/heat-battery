function pad_number(n, n_digit_max) {
    const has_n_digits = n.toString().length;
    if (has_n_digits < n_digit_max) {
        return n.toString().padStart(n_digit_max-has_n_digits, ' ');
    } else {
        return n.toString();
    };
}

function obj2pydict(obj, comment_prefix="#") {
    return comment_prefix + "\n\n" + JSON.stringify(obj, null, 4).replace("true", "True").replace("false", "False");
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

function stringify_sorted (json_obj) {
    const all_keys = Object.keys(json_obj);
    const sorted_keys = all_keys.sort();
    const sorted_obj = {};
    sorted_keys.forEach((key) => {
        sorted_obj[key] = json_obj[key];
    });
    return JSON.stringify(sorted_obj);
}

//const py_icon = React.createElement(window.dash_iconify.DashIconify, {icon:"feather:info", width: 20});

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        show_sim_inputs:  function(cellRendererData) {
            if (cellRendererData === null) {
                return dash_clientside.no_update;
            };
            if (cellRendererData.colId === 'signature_2') {
                const title = `Inputs for job ${cellRendererData.value.signature}`;
                const tabs = [
                    {
                        fileName: "sim_p.json", 
                        code: obj2pydict(cellRendererData.value.p_inputs.sim_p, "#inputs to simulation runner"), 
                        language: "python",
                        //icon: "feather:info",
                    },
                    {
                        fileName: "mesh_p.json", 
                        code: obj2pydict(cellRendererData.value.p_inputs.mesh_p, "#inputs to mesh generator"), 
                        language: "python",
                        //icon: "feather:info",
                    },

                ];
                console.log(tabs);
                //count lines in content
                //const lines = content.split('\n');
                //const n = lines.length;
                //console.log(n)
                //const n_digit_max = n.toString().length;

                //const numbered_content = lines.map((line, index) => `${pad_number(index + 1, n_digit_max)}. ${line}`).join('\n')
                return [true, title, tabs];
            };
            if (cellRendererData.colId === 'signature_3') {
                const title = `Outputs for job ${cellRendererData.value.signature}`;
                var content = JSON.stringify(cellRendererData.value.output, null, 4);
                content = "```json\n" + content + "\n```";
                return [true, title, content];
            };
        },
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
        update_graph_config: function(config, id) { 
            console.log(config);
            
            const fig_id = stringify_sorted(id);  
            config.modeBarButtonsToAdd = [[
                {
                    name: 'toggleLegend',
                    title: 'Toggle legend',
                    icon: myIcons.LegendIcon,
                    click: function(gd) {
                        var fig_div = document.getElementById(`${fig_id}`);
                        var plotly_obj = fig_div.getElementsByClassName("js-plotly-plot");
                        var value = plotly_obj[0].layout.showlegend;
                        var update = {'showlegend': !value};
                        Plotly.update(plotly_obj[0], {}, update);
                        console.log('hello from custom modbar');
                    }
                },
                {
                    name: 'toggleFullscreen',
                    title: 'Toggle fullscreen',
                    icon: myIcons.FullscreenIcon,
                    click: function(gd) {
                        var target = document.getElementById(`${fig_id}`);
                        if (document.fullscreenElement === null) {
                            target.requestFullscreen();
                        } else {
                            document.exitFullscreen();
                        };
                        console.log('hello from custom modbar');
                    }
                },
                {
                    name: 'delete', 
                    title: 'Delete graph',
                    icon: myIcons.DeleteIcon,
                    click: function(gd) {
                        // open modal with delete warning
                        const modal = document.getElementById('delete-modal');
                        modal.show();
                    }
                },
                {
                    name: 'edit', 
                    title: 'Edit graph',
                    icon: myIcons.EditIcon,
                    click: function(gd) {
                        console.log('I will edit the figure');
                    }
                }
            ]];
            return config;
        }
    }
});