
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const loadingDiv = document.getElementById('loading');

    form.style.display = 'none';
    loadingDiv.style.display = 'block';

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            window.location.href = data.results_path;
        } else {
            alert('An error occurred during processing.');
            form.style.display = 'block';
            loadingDiv.style.display = 'none';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred.');
        form.style.display = 'block';
        loadingDiv.style.display = 'none';
    });
});
