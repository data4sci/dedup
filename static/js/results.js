/* JavaScript for Results Page
   - Frame card click opens in-page modal overlay
   - Select All (stratification) toggles all .strata-filter checkboxes
*/

document.addEventListener("DOMContentLoaded", () => {
    // Modal elements (present in template)
    const modal = document.getElementById("frame-modal");
    const modalImage = document.getElementById("modal-image");
    const overlay = document.getElementById("modal-overlay");
    const closeBtn = document.getElementById("modal-close");

    function openModal(src) {
        if (!modal || !modalImage) return;
        modalImage.src = src;
        modal.style.display = "block";
        document.body.style.overflow = "hidden";
    }

    function closeModal() {
        if (!modal || !modalImage) return;
        modal.style.display = "none";
        modalImage.src = "";
        document.body.style.overflow = "";
    }

    // Open modal when any frame-card is clicked
    const frameCards = document.querySelectorAll(".frame-card");
    frameCards.forEach((card) => {
        card.style.cursor = "pointer";
        card.addEventListener("click", () => {
            const img = card.dataset.image;
            if (img) openModal(img);
        });
    });

    // Close handlers
    if (overlay) overlay.addEventListener("click", closeModal);
    if (closeBtn) closeBtn.addEventListener("click", closeModal);
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeModal();
    });

    // Select All for stratification filters
    const selectAllStrata = document.getElementById("select-all-strata");
    if (selectAllStrata) {
        selectAllStrata.addEventListener("change", (e) => {
            const checked = e.target.checked;
            document.querySelectorAll(".strata-filter").forEach((cb) => {
                cb.checked = checked;
            });
        });
    }
});
