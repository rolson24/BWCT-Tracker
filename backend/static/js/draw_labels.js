window.drawLabels = function drawLabels(lineStart, lineEnd, document) {
    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');

    var canvas_y_offset = Number((canvas.style.top).split('p')[0]);

    console.log(`${canvas.style.top.split('p')[0]}`);

    lineStart.y += canvas_y_offset;
    lineEnd.y += canvas_y_offset;

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

    var line_min_x = Math.min(lineStart.x, lineEnd.x)
    var line_min_y = Math.min(lineStart.y, lineEnd.y)

    var angle = Math.atan2(lineStart.y - lineEnd.y, lineEnd.x - lineStart.x);

    var angle_perp = Math.PI/2 - angle;
    var arrow_x = (Math.cos(angle_perp) * arrowLength)
    var arrow_y = (Math.sin(angle_perp) * arrowLength);
    // When the line is drawn bottom to top generally
    if (angle > 0 && angle <= Math.PI / 2){
        label.style.left = (lineEnd.x - labelWidth) + 'px'; // Position to the left of the endpoint
        label.style.top = (lineEnd.y- labelHeight) + 'px'; // Position above the endpoint

        midpoint_x = (lineEnd.x - lineStart.x) / 2 + line_min_x;
        midpoint_y = (lineStart.y - lineEnd.y) / 2 + line_min_y; // bottom - top / 2 + min
        
        out_left = midpoint_x + 10;
        out_top = midpoint_y;

        in_left = midpoint_x - outLabelWidth;
        in_top = midpoint_y - labelHeight;

    } else if (angle > Math.PI / 2 && angle < Math.PI) {
        label.style.left = (lineEnd.x + 10) + 'px'; // Position to the right of the endpoint
        label.style.top = (lineEnd.y- labelHeight) + 'px'; // Position above the endpoint

        midpoint_x = (lineStart.x - lineEnd.x) / 2 + line_min_x;
        midpoint_y = (lineStart.y - lineEnd.y) / 2 + line_min_y;

        out_left = midpoint_x + 10;
        out_top = midpoint_y - labelHeight;

        in_left = midpoint_x - outLabelWidth;
        in_top = midpoint_y;

    } else if (angle > -Math.PI && angle < -Math.PI / 2) {
        label.style.left = (lineEnd.x + 10) + 'px'; // Position to the right of the endpoint
        label.style.top = lineEnd.y+ 'px';

        midpoint_x = (lineStart.x - lineEnd.x) / 2 + line_min_x;
        midpoint_y = (lineEnd.y- lineStart.y) / 2 + line_min_y;

        out_left = midpoint_x - inLabelWidth;
        out_top = midpoint_y - labelHeight;

        in_left = midpoint_x + 10;
        in_top = midpoint_y;
    } else {
        label.style.left = (lineEnd.x - labelWidth) + 'px'; // Position to the left of the endpoint
        label.style.top = lineEnd.y+ 'px';

        midpoint_x = (lineEnd.x - lineStart.x) / 2 + line_min_x;
        midpoint_y = (lineEnd.y- lineStart.y) / 2 + line_min_y;

        out_left = midpoint_x - inLabelWidth;
        out_top = midpoint_y;

        in_left = midpoint_x + 10;
        in_top = midpoint_y - labelHeight;
    }
    in_label.style.left = (in_left) + 'px';
    in_label.style.top = (in_top) + 'px';
    console.log(`In label ${in_label.style.left}, ${in_label.style.top}`)

    out_label.style.left = (out_left) + 'px';
    out_label.style.top = (out_top) + 'px';
    console.log(`Out label ${out_label.style.left}, ${out_label.style.top}`)

    in_center_x = in_left + inLabelWidth/2;
    in_center_y = in_top + labelHeight/2;

    out_center_x = out_left + outLabelWidth/2;
    out_center_y = out_top + labelHeight/2;

    // in arrow
    drawArrow(ctx, in_center_x, (in_center_y - canvas_y_offset), in_center_x + arrow_x, in_center_y - canvas_y_offset + arrow_y, arrowWidth, 'blue');
    // out arrow
    drawArrow(ctx, out_center_x, out_center_y - canvas_y_offset, out_center_x - arrow_x, out_center_y - canvas_y_offset - arrow_y, arrowWidth, 'DarkGreen');

    document.querySelector('.video-container').appendChild(label);
    document.querySelector('.video-container').appendChild(out_label);
    document.querySelector('.video-container').appendChild(in_label);
    console.log(label)
    console.log(out_label)
    console.log(in_label)
    return [label, out_label, in_label];
}