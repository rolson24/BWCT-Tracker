
window.drawArrow = function drawArrow(ctx, fromx, fromy, tox, toy, arrowWidth, color) {
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
window.renderCountsTable = function renderCountsTable(countsData, document) {
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
window.renderPlot = function renderPlot(plotData, plotElementId, plotName) {
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