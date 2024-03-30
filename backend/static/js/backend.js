// This is the backend for the app.
import { 
        saveStateToLocalStorage, loadStateFromLocalStorage,
        saveCanvasState, loadCanvasState,
        saveCountsDataToLocalStorage, loadCountsDataFromLocalStorage,
        savePlotDataToLocalStorage, loadPlotDataFromLocalStorage,
        saveLabelsState, loadLabelsState
        } from "./save_page_state.js"



// Function to handle Socket.IO client reconnection
function handleSocketReconnection(socket) {
    // Attempt to reconnect with an exponential backoff strategy
    var reconnectAttempts = 0;
    var maxReconnectAttempts = 5;
    var reconnect = function() {
        if (reconnectAttempts < maxReconnectAttempts) {
            setTimeout(function() {
                if (!socket.connected) {
                    console.log('Attempting to reconnect to the WebSocket server...');
                    socket.connect();
                    reconnectAttempts++;
                    reconnect();
                }
            }, Math.pow(2, reconnectAttempts) * 1000); // Exponential backoff
        } else {
            console.log('Max reconnection attempts reached. Could not reconnect to the WebSocket server.');
        }
    };
    reconnect();
}


function drawArrow(ctx, fromx, fromy, tox, toy, arrowWidth, color) {
    var headlen = 10; // length of head in pixels
    var dx = tox - fromx;
    var dy = toy - fromy;
    var angle = Math.atan2(dy, dx);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = arrowWidth;
    ctx.beginPath();
    ctx.moveTo(fromx, fromy);
    ctx.lineTo(tox, toy);
    ctx.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
    ctx.stroke();
}


// Function to render the counts data into the table
function renderCountsTable(countsData) {
    var tableBody = document.getElementById('counts-display').querySelector('tbody');
    tableBody.innerHTML = ''; // Clear the table body
    if (countsData) {
        countsData.forEach(function(count) {
            var row = document.createElement('tr');
            var lineCell = document.createElement('td');
            var classCell = document.createElement('td');
            var directionCell = document.createElement('td');
            var countCell = document.createElement('td');

            lineCell.textContent = count.line;
            classCell.textContent = count.class;
            directionCell.textContent = count.direction;
            countCell.textContent = count.count;

            row.appendChild(lineCell);
            row.appendChild(classCell);
            row.appendChild(directionCell);
            row.appendChild(countCell);
            tableBody.appendChild(row);
        });
    }
}



// Function to render the plot using Plotly
function renderPlot(plotData, plotElementId, plotName) {
    if (plotData) {
        Plotly.newPlot(plotElementId, plotData.data, plotData.layout, {
            toImageButtonOptions: {
                filename: plotName,
                height: null,
                width: null,
                format: 'png'
            }
        });
    }
}

var processing = false;
var labels = {};  // Create an object to store the labels




function beforeUnloadListener(event) {
    event.returnValue = 'Are you sure you want to leave?';
}

// Save the state to local storage when the page is unloaded
window.addEventListener('beforeunload', function (e) {
    saveStateToLocalStorage(document, processing);
    // Save the labels state to local storage
    saveLabelsState(labels);
    // Save the canvas state to local storage
    saveCanvasState(document.getElementById('canvas'));
    // Rest of the beforeunload event...
});

window.addEventListener('beforeunload', beforeUnloadListener);

// Function to reload the video player with the current video source
function reloadVideoPlayer() {
    var videoPlayer = document.getElementById('video-player');
    if (videoPlayer) {
        var currentSrc = videoPlayer.getAttribute('src');
        videoPlayer.src = ''; // Reset the source to force reload
        videoPlayer.load(); // Load the video player without source to clear previous state
        videoPlayer.src = currentSrc; // Set the source back to the original path
        videoPlayer.load(); // Load the video player with the new source
        videoPlayer.show(); // Show the video player
    }
}

// Function to check if a video has been uploaded and reload the video player
function checkAndReloadVideoPlayer() {
    var videoPlayer = document.getElementById('video-player');
    // Check if the video player has a source and if it's not empty
    if (videoPlayer && videoPlayer.getAttribute('src')) {
        console.log("reload the video player");
        reloadVideoPlayer(); // Call the reloadVideoPlayer function if a source is present
    }
}


$(document).ready(function () {
    // Load state and plot data from local storage
    loadStateFromLocalStorage(document);
    var countsData = loadCountsDataFromLocalStorage();
    renderCountsTable(countsData);
    var countsPlotData = loadPlotDataFromLocalStorage('countsPlotData');
    renderPlot(countsPlotData, 'counts-plot', 'counts_by_line');
    var crossingsPlotData = loadPlotDataFromLocalStorage('crossingsPlotData');
    renderPlot(crossingsPlotData, 'crossings-plot', 'counts_per_hour');
    var trackPlotData = loadPlotDataFromLocalStorage('trackPlotData');
    renderPlot(trackPlotData, 'track-plot', 'tracks_overlay');
    var personVolumePlotData = loadPlotDataFromLocalStorage('personVolumePlotData');
    renderPlot(personVolumePlotData, 'person-volume-plot', 'person_volume');

    var drawing = false;
    var lineStart = null;

    $('#video-player, #canvas').hide();

    // Call checkAndReloadVideoPlayer when the page is ready
    checkAndReloadVideoPlayer();

    // Load the labels state when the page is ready
    loadLabelsState(document);

    // Load the canvas state when the page is ready
    loadCanvasState(document.getElementById('canvas'));

    $('.custom-file-input').on("change", function () {
        var fileName = $(this).val().split("\\").pop();
        $(this).siblings('.custom-file-label').addClass("selected").html(fileName);
    });
    var video_width;
    var video_height;

    $('#video-upload-button').on('click', function(e) {
        // Prevent the default form submission if the button is part of a form
        e.preventDefault();

        // Call the openFileDialog function exposed by the preload script
        window.electronAPI.openFileDialog().then(result => {
            // Assuming openFileDialog is adjusted to return a Promise that resolves with the file selection result
            console.log("get filenames")
            console.log(result)
            // You can now do something with the selected file paths, like sending them to your Flask backend
            $.ajax({
                url: '/receive-file-paths',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ filenames: result }),
                success: function(response) {
                    console.log('Server response:', response);
                    $('#draw-button').show();
                    $('#process-button').show();  // Show the "Process Video" button
                    $('#reprocess-button').show();  // Show the "Reprocess Counts" button
                    $('#clear-lines-button').show();  // Show the "Clear Lines" button
                    // var video_name = response.filename.split('.')[0];
                    // var video_filename = response.filename;

                    $('#video-player').attr('src', `/stream_video`, 'type', 'video/mp4').css({ width: '100%', height: '100%' }).prop('controls', true).on('loadedmetadata', function () {
                        var canvas = $('#canvas')[0];
                        var ctx = canvas.getContext('2d');
                        video_width = this.videoWidth;
                        video_height = this.videoHeight;
                        var aspectRatio = this.videoHeight / this.videoWidth;
                        var canvasHeight = this.clientWidth * aspectRatio;
                        console.log(`Canvas height: ${canvasHeight}`)
                        var yOffset = (this.clientHeight - canvasHeight) / 2
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        $('#canvas').css("top", yOffset).attr({ width: this.clientWidth, height: canvasHeight }).css('pointer-events', 'none').show();
                    }).show();

                    $('#reprocess-button').prop('disabled', true)

                    $('#download-counts-link').attr('href', '/download_counts');
                    $('#download-processed-video-link').attr('href', '/download_processed_video');
                },
                error: function(xhr, status, error) {
                    console.error('Error sending filenames:', error);
                }
            })
        }).catch(err => {
            console.error('Error opening file dialog:', err);
        });
    });

    function getBasename(filePath) {
        // Use a regular expression to extract the part after the last slash
        return filePath.replace(/^.*[\\\/]/, '');
    }
    const getMeta = (url) =>
        new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = (err) => reject(err);
            img.src = url;
        });


    $('#raw-data-button').on('click', function(e) {
        // Prevent the default form submission if the button is part of a form
        e.preventDefault();

        // Call the openFileDialog function exposed by the preload script
        window.electronAPI.openFileDialog().then(result => {
            // Assuming openFileDialog is adjusted to return a Promise that resolves with the file selection result
            console.log("get filenames")
            console.log(result)
            // You can now do something with the selected file paths, like sending them to your Flask backend
            $.ajax({
                url: '/receive-raw-tracks-file-path',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ filenames: result }),
                success: function(response) {
                    console.log('Server response:', response);
                    $('#draw-button').show();
                    $('#process-button').show();  // Show the "Process Video" button
                    $('#reprocess-button').show();  // Show the "Reprocess Counts" button
                    $('#clear-lines-button').show();  // Show the "Clear Lines" button
                    // var video_name = response.filename.split('.')[0];
                    // var video_filename = response.filename;

                    video_name = getBasename(result[0]).split('.')[0];
                    image_name = `/static/outputs/${video_name}/run_1/middle_frame.jpg`
                    console.log(`image name: ${image_name}`)

                    $('#video-player').css({ width: '100%', height: '100%' }).prop('controls', false).show();
                    var videoPlayer = document.getElementById("video-player")


                    ;(async() => {
                        const img = await getMeta(image_name);
                        console.dir(img.naturalHeight + ' ' + img.naturalWidth);
                        video_width = img.naturalWidth;
                        video_height = img.naturalHeight;

                        var canvas = $('#canvas')[0];
                        var ctx = canvas.getContext('2d');
                        var aspectRatio = video_height / video_width;
                        console.log(`Video Aspect Ratios: ${aspectRatio}`)
                        var canvasHeight = videoPlayer.clientWidth * aspectRatio;
                        console.log(`Canvas height: ${canvasHeight}`)
                        var yOffset = (videoPlayer.clientHeight - canvasHeight) / 2
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        $('#canvas').css("top", yOffset).attr({ width: videoPlayer.clientWidth, height: canvasHeight }).css('pointer-events', 'none').show();
                        console.log("setup canvas")
                    })();


                    // Use the Fetch API to make a request to the file's URL
                    fetch(image_name)
                        .then(response => {
                            // Check if the file is accessible
                            if(response.ok) {
                                console.log('The file is accessible.');
                                videoPlayer.poster = image_name
                            } else {
                                console.log('The file is not accessible.', response.statusText);
                                // Handle the case where the file is not accessible
                            }
                        })
                        .catch(error => {
                            // Handle any errors that occur during the fetch
                            console.error('Error fetching the file:', error);
                        });



                    $('#process-button').prop('disabled', true)

                    $('#download-counts-link').attr('href', '/download_counts');
                    $('#download-processed-video-link').attr('href', '/download_processed_video');
                },
                error: function(xhr, status, error) {
                    console.error('Error sending filenames:', error);
                }
            })
        }).catch(err => {
            console.error('Error opening file dialog:', err);
        });
    });


    $('#upload-day-night-form').on('submit', function(e) {
        e.preventDefault(); // Prevent default form submission behavior

        var formData = new FormData(this); // Create a FormData object from the form
        $.ajax({
            url: '/upload_day_night', // Your Flask route
            type: 'POST',
            data: formData,
            contentType: false, // Necessary for file upload
            processData: false, // Necessary for file upload
            success: function(response) {
                alert(response.message); // Show success message
            },
            error: function(xhr) {
                alert('An error occurred'); // Show error message
            }
        });
    });

    $('#draw-button').on('click', function () {
        console.log('Draw button clicked');
        drawing = !drawing;
        $('#video-player').prop('controls', !drawing);
        $('#canvas').css('pointer-events', drawing ? 'auto' : 'none');
        $(this).text(drawing ? 'Stop Drawing' : 'Draw Lines');
    });

    $('#clear-lines-button').on('click', function () {
        var ctx = $('#canvas')[0].getContext('2d');
        ctx.clearRect(0, 0, $('#canvas')[0].width, $('#canvas')[0].height);
        $.post('/clear_lines', function (response) {
            // alert(response.message);
        });
        for (var label in labels) {
            label_obj = labels[label];
            if (Array.isArray(label_obj)) {
                for (var dir_label in label_obj) {
                    console.log(`dir_label: ${label_obj[dir_label]} type: ${typeof label_obj[dir_label]}`)
                    label_obj[dir_label].remove()
                }
            } else {
                label_obj.remove()
            }
        }
        labels = {};
        labels['Out'] = [];
        labels['In'] = [];
    });
    const forceReflow = (element) => {
        $('#video-upload-button').focus()
    };

    labels['Out'] = [];
    labels['In'] = [];
    $('#canvas').on('click', function (e) {
        if (drawing) {
            var rect = this.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
            if (lineStart) {
                var ctx = this.getContext('2d');
                ctx.beginPath();
                ctx.moveTo(lineStart.x, lineStart.y);
                ctx.lineTo(x, y);
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 5;
                ctx.stroke();
                var video = document.getElementById('video-player');
                var canvas = document.getElementById('canvas');
                var scaleX = video_width / canvas.width;
                var scaleY = video_height / canvas.height;
                // var scaleX = 1;
                // var scaleY = 1;
                console.log(`scaleX: ${scaleX}, scaleY: ${scaleY}`)
                console.log(`Line drawn: (${Math.round(lineStart.x * scaleX)},${Math.round(lineStart.y * scaleY)}) (${Math.round(x * scaleX)},${Math.round(y * scaleY)})`)
                $.post('/coordinates', { line: `(${Math.round(lineStart.x * scaleX)},${Math.round(lineStart.y * scaleY)}) (${Math.round(x * scaleX)},${Math.round(y * scaleY)})` }, function (response) {
                    // alert(response.message);
                });
                // Create a label element for the line
                var label = document.createElement('span');
                label.textContent = 'Line ' + ((Object.keys(labels).length - 2));  // Use the number of lines as the label
                label.style.position = 'absolute';
                label.style.color = 'red';
                label.style.fontSize = '15px';
                label.style.fontWeight = 'bold';
                label.style.pointerEvents = 'none';  // Prevent the label from being interacted with

                var out_label = document.createElement('span');
                out_label.textContent = 'Out';  // Use the number of lines as the label
                out_label.style.position = 'absolute';
                out_label.style.color = 'DarkGreen';
                out_label.style.fontSize = '15px';
                out_label.style.fontWeight = 'bold';
                out_label.style.pointerEvents = 'none';  // Prevent the label from being interacted with

                var in_label = document.createElement('span');
                in_label.textContent = 'In';  // Use the number of lines as the label
                in_label.style.position = 'absolute';
                in_label.style.color = 'blue';
                in_label.style.fontSize = '15px';
                in_label.style.fontWeight = 'bold';
                in_label.style.pointerEvents = 'none';  // Prevent the label from being interacted with

                var labelWidth = 50;
                var outLabelWidth = 30;
                var inLabelWidth = 20;
                var labelHeight = 20;

                var arrowLength = 40;
                var arrowWidth = 2;

                var canvas_y_offset = Number((canvas.style.top).split('p')[0]);
                console.log(`${canvas.style.top.split('p')[0]}`);

                lineStart.y += canvas_y_offset;
                y += canvas_y_offset;

                var line_min_x = Math.min(lineStart.x, x)
                var line_min_y = Math.min(lineStart.y, y)

                // Calculate the angle of the line
                var angle = Math.atan2(lineStart.y - y, x - lineStart.x);
                // Determine the direction of the line and position the label accordingly
                console.log('Angle:', angle);
                if (angle > 0 && angle < Math.PI / 2) {
                    // Line is drawn from bottom left to upper right
                    console.log('Line is drawn from bottom left to upper right');
                    label.style.left = (x - labelWidth) + 'px'; // Position to the left of the endpoint
                    label.style.top = (y - labelHeight) + 'px'; // Position above the endpoint

                    midpoint_x = (x - lineStart.x) / 2 + line_min_x;
                    midpoint_y = (lineStart.y - y) / 2 + line_min_y;
                    arrow_midpoint_y = midpoint_y - canvas_y_offset;
                    angle_new = Math.PI/2 - angle
                    console.log(`angle new: ${angle_new}`)
                    arrow_x = Math.abs(Math.cos(angle_new) * arrowLength);
                    arrow_y = Math.abs(Math.sin(angle_new) * arrowLength);
                    console.log(`arrow x: ${arrow_x}, arrow y: ${arrow_y}`)

                    out_left = midpoint_x + 10;
                    out_top = midpoint_y;

                    in_left = midpoint_x - outLabelWidth;
                    in_top = midpoint_y - labelHeight;

                    in_center_x = in_left + inLabelWidth/2;
                    in_center_y = in_top + labelHeight/2;

                    out_center_x = out_left + outLabelWidth/2;
                    out_center_y = out_top + labelHeight/2;

                    in_label.style.left = (in_left) + 'px'          // Position to the right of the line
                    in_label.style.top = (in_top) + 'px'  // Position below the line
                    console.log(`In label ${in_label.style.left}, ${in_label.style.top}`)
                    // in arrow
                    drawArrow(ctx, in_center_x, (in_center_y - canvas_y_offset), in_center_x + arrow_x, in_center_y - canvas_y_offset + arrow_y, arrowWidth, 'blue');

                    out_label.style.left = (out_left) + 'px'    // Position to the left of the line
                    out_label.style.top = (out_top) + 'px'   // Position above the line
                    console.log(`Out label ${out_label.style.left}, ${out_label.style.top}`)
                    // out arrow
                    drawArrow(ctx, out_center_x, out_center_y - canvas_y_offset, out_center_x - arrow_x, out_center_y - canvas_y_offset - arrow_y, arrowWidth, 'DarkGreen');

                } else if (angle > Math.PI / 2 && angle < Math.PI) {
                    // Line is drawn from bottom right to top left
                    console.log('Line is drawn from bottom right to top left');
                    label.style.left = (x + 10) + 'px'; // Position to the right of the endpoint
                    label.style.top = (y - labelHeight) + 'px'; // Position below the endpoint

                    midpoint_x = (lineStart.x - x) / 2 + line_min_x;
                    midpoint_y = (lineStart.y - y) / 2 + line_min_y;
                    arrow_midpoint_y = midpoint_y - canvas_y_offset;

                    angle_new = Math.PI/2 - angle
                    console.log(`angle new: ${angle_new}`)
                    arrow_x = Math.abs(Math.cos(angle_new) * arrowLength);
                    arrow_y = Math.abs(Math.sin(angle_new) * arrowLength);
                    console.log(`arrow x: ${arrow_x}, arrow y: ${arrow_y}`)

                    out_left = midpoint_x + 10;
                    out_top = midpoint_y - labelHeight;

                    in_left = midpoint_x - outLabelWidth;
                    in_top = midpoint_y;

                    in_center_x = in_left + inLabelWidth/2;
                    in_center_y = in_top + labelHeight/2;

                    out_center_x = out_left + outLabelWidth/2;
                    out_center_y = out_top + labelHeight/2;

                    in_label.style.left = (in_left) + 'px' // Position to the right of the line
                    in_label.style.top = (in_top) + 'px' // Position above the line
                    console.log(`In label ${in_label.style.left}, ${in_label.style.top}`)
                    // in arrow
                    drawArrow(ctx, in_center_x, in_center_y - canvas_y_offset, in_center_x + arrow_x, in_center_y - canvas_y_offset - arrow_y, arrowWidth, 'blue');

                    out_label.style.left = (out_left) + 'px' // Position to the left of the line
                    out_label.style.top = (out_top) + 'px'  // Position below the line
                    // out arrow
                    drawArrow(ctx, out_center_x, out_center_y - canvas_y_offset, out_center_x - arrow_x, out_center_y - canvas_y_offset + arrow_y, arrowWidth, 'DarkGreen');

                } else if (angle > -Math.PI && angle < -Math.PI / 2) {
                    // Line is drawn from top right to bottom left
                    console.log('Line is drawn from top right to bottom left');
                    label.style.left = (x + 10) + 'px'; // Position to the right of the endpoint
                    label.style.top = y + 'px';

                    midpoint_x = (lineStart.x - x) / 2 + line_min_x;
                    midpoint_y = (y - lineStart.y) / 2 + line_min_y;
                    arrow_midpoint_y = midpoint_y - canvas_y_offset;

                    angle_new = Math.PI/2 - angle
                    console.log(`angle new: ${angle_new}`)
                    arrow_x = Math.abs(Math.cos(angle_new) * arrowLength);
                    arrow_y = Math.abs(Math.sin(angle_new) * arrowLength);
                    console.log(`arrow x: ${arrow_x}, arrow y: ${arrow_y}`)

                    out_left = midpoint_x - inLabelWidth;
                    out_top = midpoint_y - labelHeight;

                    in_left = midpoint_x + 10;
                    in_top = midpoint_y;

                    in_center_x = in_left + inLabelWidth/2;
                    in_center_y = in_top + labelHeight/2;

                    out_center_x = out_left + outLabelWidth/2;
                    out_center_y = out_top + labelHeight/2;

                    in_label.style.left = (in_left) + 'px' // Position to the left of the line
                    in_label.style.top = (in_top) + 'px' // Position above the line
                    console.log(`In label ${in_label.style.left}, ${in_label.style.top}`)
                    // in arrow
                    drawArrow(ctx, in_center_x, in_center_y - canvas_y_offset, in_center_x - arrow_x, in_center_y - canvas_y_offset - arrow_y, arrowWidth, 'blue');

                    out_label.style.left = (out_left) + 'px'           // Position to the right of the line
                    out_label.style.top = (out_top) + 'px'   // Position below the line
                    // out arrow
                    drawArrow(ctx, out_center_x, out_center_y - canvas_y_offset, out_center_x + arrow_x, out_center_y - canvas_y_offset + arrow_y, arrowWidth, 'DarkGreen');


                } else {
                    // Line is drawn from top left to bottom right
                    console.log('Line is drawn from top left to bottom right');
                    label.style.left = (x - labelWidth) + 'px'; // Position to the left of the endpoint
                    label.style.top = y + 'px';

                    midpoint_x = (x - lineStart.x) / 2 + line_min_x;
                    midpoint_y = (y - lineStart.y) / 2 + line_min_y;
                    arrow_midpoint_y = midpoint_y - canvas_y_offset;

                    angle_new = Math.PI/2 - angle
                    console.log(`angle new: ${angle_new}`)
                    arrow_x = Math.abs(Math.cos(angle_new) * arrowLength);
                    arrow_y = Math.abs(Math.sin(angle_new) * arrowLength);
                    console.log(`arrow x: ${arrow_x}, arrow y: ${arrow_y}`)

                    out_left = midpoint_x - inLabelWidth;
                    out_top = midpoint_y;

                    in_left = midpoint_x + 10;
                    in_top = midpoint_y - labelHeight;

                    in_center_x = in_left + inLabelWidth/2;
                    in_center_y = in_top + labelHeight/2;

                    out_center_x = out_left + outLabelWidth/2;
                    out_center_y = out_top + labelHeight/2;

                    in_label.style.left = (in_left) + 'px' // Position to the left of the line
                    in_label.style.top = (in_top) + 'px' // Position below the line
                    console.log(`In label ${in_label.style.left}, ${in_label.style.top}`)
                    // in arrow
                    drawArrow(ctx, in_center_x, in_center_y - canvas_y_offset, in_center_x - arrow_x, in_center_y - canvas_y_offset + arrow_y, arrowWidth, 'blue');

                    out_label.style.left = (out_left) + 'px' // Position to the right of the line
                    out_label.style.top = (out_top) + 'px'  // Position above the line
                    // out arrow
                    drawArrow(ctx, out_center_x, out_center_y - canvas_y_offset, out_center_x + arrow_x, out_center_y - canvas_y_offset - arrow_y, arrowWidth, 'DarkGreen');

                }
                document.querySelector('.video-container').appendChild(label);
                document.querySelector('.video-container').appendChild(out_label);
                document.querySelector('.video-container').appendChild(in_label);
                labels[label.textContent] = label;  // Store the label in the labels object
                labels["Out"].push(out_label);
                labels["In"].push(in_label);
                lineStart = null;


                // label.hide().show(0);
                // label.style.display = 'none';
                // label.style.display = 'block';

                // out_label.hide().show(0);
                // in_label.hide().show(0);
                // canvas.hide().show(0);

                // Apply to the body or specific element needing refresh
                forceReflow($('#video-container'));
            } else {
                lineStart = { x: x, y: y };
            }
        }
        forceReflow($('#video-container'));


        document.getElementById("thisisfucked").clear
    });


    console.log('JavaScript code run');  // Log when the JavaScript code is run

    $('#process-button').on('click', function () {
        if (processing == false) {
            var confirmProcess = confirm("Are you sure you want to process the full video? This could take several hours. Remember if you just want to process again with new lines, you can use the 'Reprocess Counts' button");
            if (!confirmProcess) {
                return; // User clicked 'No', exit the function
            }

            $('#process-button').prop('disabled', true)
            $('#reprocess-button').prop('disabled', true)

            processing = true
            $('#processing-progress').css('width', 0 + '%').attr('aria-valuenow', 0);
            // var formData = new FormData(document.getElementById('upload-form'));
            // var filename = formData.get('video').name;  // Assuming 'video' is the name attribute of your file input
            var saveVideo = $('#save-video').val(); // Get the dropdown value

            // formData.append('save_video', saveVideo); // Append the choice to your form data
            // formData.append('filename', filename); // Append the filename to your form data
            $('#processing-loader').show()
            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',  // Indicate that you're sending JSON data
                },
                body: JSON.stringify({ save_video: saveVideo })  // Convert your data into a JSON string

            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Video processing started:', data);
                })
                .catch(error => {
                    console.error('There has been a problem with your fetch operation:', error);
                });
        }
    });



    $('#reprocess-button').on('click', function () {
        // var formData = new FormData(document.getElementById('upload-form'));
        // var filename = formData.get('video').name;  // Get the filename from the form data
        $('#processing-progress').css('width', '0%').attr('aria-valuenow', 0);  // Reset the progress bar
        $('#processing-loader').show()

        fetch('/reprocess', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // body: JSON.stringify({ filename: filename })  // Pass the filename as JSON
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log('Video reprocessing started:', data);
            })
            .catch(error => {
                console.error('There has been a problem with your fetch operation:', error);
            });
    });


    document.getElementById('download-counts-button').addEventListener('click', () => {
        window.electronAPI.saveCountsFile();
    });

    $('#download-raw-data-button').on('click', function () {
        window.electronAPI.saveRawTracksFile();
    });

    $('#download-lines-button').on('click', function () {
        window.electronAPI.saveLineCrossingsFile();
    });

    $('#download-processed-video-button').on('click', function () {
        window.electronAPI.saveProcessedVideoFile();

    });

    $('#download-plots-button').on('click', function () {
        window.electronAPI.savePlots();
        $('#download-plots-button').prop('disabled', true);

    });



    var socket = io.connect('http://' + document.domain + ':' + location.port);
    socket.on('disconnect', function() {
        console.log('Disconnected from WebSocket server.');
        handleSocketReconnection(socket);
    });
    socket.on('connect_error', function(error) {
        console.log('Connection error:', error);
        handleSocketReconnection(socket);
    });
    socket.on('progress', function (msg) {
        console.log('Processing progress:', msg);
        $('#processing-progress').css('width', msg.data + '%').attr('aria-valuenow', msg.data);
        $('#estimated-time').text('Estimated time remaining: ' + msg.time)
    });
    socket.on('plot-download-ready', function() {
        console.log('Ready to download plots');
        $('#download-plots-button').prop('disabled', false);

    })
    var video_name;
    // Listen for the video_processed event
    socket.on('video_processed', function (msg) {
        console.log('Video processed event received:', msg);  // Log the event data
        $('#processing-progress').css('width', '100%').attr('aria-valuenow', 100);
        $('#estimated-time').text('Estimated time remaining: 00:00:00')
        $('#processing-loader').hide()
        processing = false
        $('#process-button').prop('disabled', false);
        $('#reprocess-button').prop('disabled', false);




        // Make an AJAX request to get the counts data
        fetch('/get_counts')
            .then(response => response.json())
            // Inside the fetch success callback
            .then(data => {
                // Parse the counts data into an array of objects
                // var lines = data.counts.split('\n\n');
                // var countsData = [];
                countsData = data.countsData;
                var currentLine = 0;
                var avgFPS;
                video_name = data.filename
                // Parse the JSON string into a JavaScript object
                var fig = JSON.parse(data.plot);
                console.log(`plotly figure ${fig.data}`);
                // Use Plotly to render the graph
                var countsPlotData = { data: fig.data, layout: fig.layout };
                renderPlot(countsPlotData, 'counts-plot', video_name + '_counts_by_line');
                saveCountsDataToLocalStorage(countsData); // Save the counts data to local storage
                savePlotDataToLocalStorage(countsPlotData, 'countsPlotData');



                // Get the table body element
                var tableBody = document.getElementById('counts-display').querySelector('tbody');

                // Clear the table body
                tableBody.innerHTML = '';

                // Insert each count as a row in the table
                countsData.forEach(count => {
                    var row = document.createElement('tr');
                    var lineCell = document.createElement('td');
                    var classCell = document.createElement('td');
                    var directionCell = document.createElement('td');
                    var countCell = document.createElement('td');

                    lineCell.textContent = count.line;
                    classCell.textContent = count.class;
                    directionCell.textContent = count.direction;
                    countCell.textContent = count.count;

                    var className = 'class-' + count.class.replace(/\s+/g, '-');
                    row.classList.add(className);

                    row.appendChild(lineCell);
                    row.appendChild(classCell);
                    row.appendChild(directionCell);
                    row.appendChild(countCell);
                    tableBody.appendChild(row);
                });

                // Insert the average FPS into the table
                if (avgFPS) {
                    var row = document.createElement('tr');
                    var cell = document.createElement('td');
                    cell.colSpan = 4;
                    cell.textContent = 'Average FPS: ' + avgFPS;
                    row.appendChild(cell);
                    tableBody.appendChild(row);
                }
                $('#download-raw-data-button').show(); // Show the "Download Raw Data" button
                $('#download-lines-button').show();  // Show the "Download Line Crossings" button
                $('#download-counts-button').show();  // Show the "Download Counts Output" button
                $('#download-processed-video-button').show();  // Show the "Download Processed Video" button
                $('#download-plots-button').show();  // Show the "Download Processed Video" button

            })
            .catch(error => {
                console.error('Error fetching counts data:', error);
            });
        console.log('get crossings data');

        var videoFPS = 30;  // Set the video FPS to 30
        fetch('/get_crossings_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // Include the fps in the body of the request
            body: JSON.stringify({ fps: videoFPS })
        })
            .then(response => response.json())
            .then(data => {
                var fig = JSON.parse(data.plot);

                // Use Plotly to render the graph
                var crossingsPlotData = { data: fig.data, layout: fig.layout };
                renderPlot(crossingsPlotData, 'crossings-plot', video_name + '_counts_per_hour');
                savePlotDataToLocalStorage(crossingsPlotData, 'crossingsPlotData');


            });

        fetch('/get_person_volume_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // Include the fps in the body of the request
            body: JSON.stringify({ fps: videoFPS })
        })
            .then(response => response.json())
            .then(data => {
                var fig = JSON.parse(data.plot);

                // Use Plotly to render the graph
                var personVolumePlotData = { data: fig.data, layout: fig.layout };
                renderPlot(personVolumePlotData, 'person-volume-plot', video_name + '_person_volume');
                savePlotDataToLocalStorage(personVolumePlotData, 'personVolumePlotData');
            });

        fetch('/get_tracks', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(response => response.json())
            .then(data => {
                // Parse the JSON string into a JavaScript object
                var fig = JSON.parse(data.fig_json);
                // var video_name = msg.filename.split('.')[0]

                // Use Plotly to render the graph
                var trackPlotData = { data: fig.data, layout: fig.layout };
                renderPlot(trackPlotData, 'track-plot', video_name + '_tracks_overlay');
                savePlotDataToLocalStorage(trackPlotData, 'trackPlotData');

            })
            .catch(error => console.error('Error fetching the Plotly graph:', error));

    });



});