function uploadVideo() {
    const fileInput = document.getElementById('videoInput');
    const file = fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('video', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    }).then(response => response.text())
      .then(html => document.body.innerHTML = html)
      .catch(error => console.error('Error:', error));
}

function updateSystemStatus() {
    fetch('/system_status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('cpuUsage').textContent = `CPU: ${data.cpu.toFixed(1)}%`;
            document.getElementById('gpuUsage').textContent = `GPU: ${data.gpu}%`;
            document.getElementById('memUsage').textContent = `RAM: ${data.mem.toFixed(1)}%`;
        });
}

setInterval(updateSystemStatus, 2000);

// Initial blank canvas
const canvas = document.getElementById('videoCanvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = '#1e1e1e';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.fillStyle = '#c8c8c8';
ctx.font = '20px Helvetica';
ctx.fillText('Upload a video or start webcam', 150, 240);