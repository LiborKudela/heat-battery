/**
 * @license
 * Custom cell renderers for ag-grid in jobs overview table
 */

var dagcomponentfuncs = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

var menu_text_style = {
    fontSize: '16px',
    fontWeight: 'bold',
};
var menu_icon_size = 20;

menuItem_Header = function (props, text) {
    return React.createElement(
        window.dash_bootstrap_components.DropdownMenuHeader,
        
        React.createElement(
            window.dash_bootstrap_components.Label,
            null,
            text
        )
    );
}

menuItem_OpenResult = function (props) {
    var project_name = props.colDef.cellRendererParams.project_name;
    
    return React.createElement(
        window.dash_bootstrap_components.DropdownMenuItem,
        {
            disabled: props.data['progress'] == 0.0,
            href: `/${project_name}/result-data?signature=${props.value}`,
            className: "opacity-100",
        },
        React.createElement(
            'div',
            null,
            React.createElement(
                window.dash_iconify.DashIconify,
                {
                    icon: 'oui:vis-query-sql',
                    width: menu_icon_size,
                    style: {
                        marginRight: '5px',
                    }
                }
            ),
            React.createElement(
                window.dash_bootstrap_components.Label,
                {style: menu_text_style},
                'Go to result detail'
            )
        )
    )
}

menuItem_ViewInputs = function (props) {
    return React.createElement(
        window.dash_bootstrap_components.DropdownMenuItem,
        null,
        React.createElement(
            'div',
            {
                id: JSON.stringify({
                    type: 'view-inputs-button',
                    index: props.data['signature']
                }),
                n_clicks: 0,
                onClick: (event) => {
                    props.setData(
                        { 
                            signature: props.data['signature'],
                            p_inputs: props.data['p_inputs']
                        }
                    );
                }
            },
            React.createElement(
                'div',
                null,
                React.createElement(
                    window.dash_iconify.DashIconify,
                    {
                        icon: 'oui:input-output',
                        width: menu_icon_size,
                        style: {
                            marginRight: '5px',
                        }
                    }
                ),
                React.createElement(
                    window.dash_bootstrap_components.Label,
                    {style: menu_text_style},
                    "View inputs/outputs"
                )
            )
        )
    );
}


menuItem_Divider = function (props) {
    return React.createElement(
        window.dash_bootstrap_components.DropdownMenuItem,
        {divider: true},
    );
}

menuItem_Header = function (props, text) {
    return React.createElement(
        window.dash_bootstrap_components.DropdownMenuItem,
        {header: true},
        text
    );
}

dagcomponentfuncs.actionsMenu = function (props) {
    return React.createElement(
        window.dash_bootstrap_components.DropdownMenu,
        {
            label: 'Actions',
            size: 'sm',
            direction: 'right',
            color: 'secondary',
            
        },
        menuItem_Header(props, "Database actions"),
        menuItem_OpenResult(props),
        menuItem_ViewInputs(props),
        menuItem_Divider(props),
        menuItem_Header(props, "Simulation actions"),
        React.createElement(
            window.dash_bootstrap_components.DropdownMenuItem,
            null,
            "item 2"
        )
    );
};

dagcomponentfuncs.viewOutputsBtn = function (props) {
    return React.createElement(
        window.dash_bootstrap_components.Button,
        {
            children: 'View outputs',
            id: JSON.stringify({
                type: 'view-outputs-button',
                index: props.data['signature']
            }),
            color: 'secondary',
            size: 'sm',
            style: {display: 'inline-block', margin: 'auto'},
            n_clicks: 0,
            onClick: () => {
                props.setData({
                    signature: props.data['signature'],
                    output: props.data['output']
                });
                
            }
        }
    );
};

dagcomponentfuncs.statusBadge = function (props) {
    // switch for badge type
    var color;
    var status_text;
    if (props.value.startsWith('SCHEDULED')) {
        color = 'primary';
        status_text = 'SCHEDULED';
    } else if (props.value.startsWith('RUNNING')) {
        color = 'warning';
        status_text = 'RUNNING';
    } else if (props.value.startsWith('COMPLETED')) {
        color = 'success';
        status_text = 'COMPLETED';
    } else if (props.value.startsWith('FAILED')) {
        color = 'danger';
        status_text = 'FAILED';
    } else if (props.value.startsWith('INTERRUPTED')) {
        color = 'dark';
        status_text = 'INTERRUPTED';
    } else {
        color = 'dark';
        status_text = 'UNKNOWN';
    }

    return React.createElement(
        window.dash_bootstrap_components.Badge,
        {
            children: status_text,
            color: color,
            style: {display: 'inline-block', margin: 'auto', width: '100px'},
        }
    );
};

dagcomponentfuncs.progressBar = function (props) {
    
    console.log(props.data['progress'] < 100);
    var progressElement = React.createElement(
        
        window.dash_bootstrap_components.Progress, 
        {
            value: props.data['progress'],
            style: {width: '100%'},
            animated: true,
            striped: true,
            color: (props.data['progress'] < 100) ? 'danger' : 'success',
            id: `progress-bar-${props.data['signature']}`
        },
    );

    return React.createElement(
        'div',
        {
            style: {height: '100%', display: 'flex', alignItems: 'center'},
            children: progressElement
        }
    );
};

function secondsToReadable(totalSeconds, appendText = '', subsecondsText = '') {
    if (totalSeconds == undefined) {
        return 'N/A';
    }

    const weeks = Math.floor(totalSeconds / (7 * 24 * 60 * 60));
    const days = Math.floor((totalSeconds % (7 * 24 * 60 * 60)) / (24 * 60 * 60));
    const hours = Math.floor((totalSeconds % (24 * 60 * 60)) / (60 * 60));
    const minutes = Math.floor((totalSeconds % (60 * 60)) / 60);
    const seconds = Math.floor(totalSeconds % 60);
    
    if (weeks > 0) {
        return `${weeks}w ${days}d ${appendText}`;
    } else if (days > 0) {
        return `${days}d ${hours}h ${appendText}`;
    } else if (hours > 0) {
        return `${hours}h ${minutes}m ${appendText}`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds}s ${appendText}`;
    } else if (seconds > 0) {
        return `${seconds}s ${appendText}`;
    } else {
        return subsecondsText;
    }
}

dagcomponentfuncs.remainingTime = function (props) { 
    return React.createElement(
        'div',
        {
            children: secondsToReadable(props.value, 'remains', 'none')
        }
    );
};

dagcomponentfuncs.elapsedTime = function (props) { 
    return React.createElement(
        'div',
        {
            children: secondsToReadable(props.value, '', 'none')
        }
    );
};

dagcomponentfuncs.inserted = function (props) { 
    const update_time = new Date(props.value)
    const diff = Date.now() - update_time.getTime();
    const totalSeconds = Math.floor(diff / 1000);
    return React.createElement(
        'div',
        {
            children: secondsToReadable(totalSeconds, 'ago', 'now')
        }
    );
};

dagcomponentfuncs.lastUpdated = function (props) { 
    const update_time = new Date(props.value)
    const diff = Date.now() - update_time.getTime();
    const totalSeconds = Math.floor(diff / 1000);
    return React.createElement(
        'div',
        {
            children: secondsToReadable(totalSeconds, 'ago', 'now')
        }
    );
};

dagcomponentfuncs.lastCheckpoint = function (props) { 
    if (props.value == undefined) {
        return 'N/A';
    }
    const update_time = new Date(props.value)
    const diff = Date.now() - update_time.getTime();
    const totalSeconds = Math.floor(diff / 1000);
    return React.createElement(
        'div',
        {
            children: secondsToReadable(totalSeconds, 'ago', 'now')
        }
    );
};

dagcomponentfuncs.createdByAvatar = function (props) {
    const avatar_map = {
        'github_username': 'https://github.com/',
        'email': 'https://unavatar.io/',
        'x_handle': 'https://unavatar.io/x/',
    }
    if (props.value == undefined) { 
        var avatar_props = {
            name: 'N/A',
            color: 'black',
            radius: 'xl',
        }
    } else if (props.value?.force_avatar_key != undefined) {
        if (props.value[props.value?.force_avatar_key] != undefined) {
            var avatar_props = {
                src: avatar_map[props.value.force_avatar_key] + props.value[props.value.force_avatar_key],
                radius: 'xl',
            }
        } else {
            var avatar_props = {
                name: 'N/A',
                color: 'black',
                radius: 'xl',
            }
        }
    } else {
        for (const key in ['github_username', 'email', 'x_handle']) {
            if (props.value[key] != null) {
                var avatar_props = {
                    src: avatar_map[key] + props.value[key],
                    radius: 'xl',
                }
                break;
            }
        
        }
        
    }
    if (props.value.link != null) {
        return React.createElement(
            'a',
            {href: props.value.link, target: '_blank'},
            React.createElement(
                window.dash_mantine_components.Avatar,
                avatar_props,
            )
        );
    } else {
        return React.createElement(
            window.dash_mantine_components.Avatar,
            avatar_props,
        );
    }
};
