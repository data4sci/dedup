
document.addEventListener('DOMContentLoaded', function() {
    const frameCards = document.querySelectorAll('.frame-card');
    const filterCheckboxes = document.querySelectorAll('.strata-filter');

    filterCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', filterFrames);
    });

    function filterFrames() {
        const activeFilters = Array.from(document.querySelectorAll('.strata-filter:checked')).map(cb => cb.value);

        frameCards.forEach(card => {
            const cardStrata = card.dataset.strata.split(',');
            const isVisible = activeFilters.some(filter => cardStrata.includes(filter));
            card.style.display = isVisible ? 'block' : 'none';
        });
    }
});
