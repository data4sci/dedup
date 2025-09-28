/* JavaScript for Results Page
   - Frame card click opens in-page modal overlay
   - Select All (stratification) toggles all .strata-filter checkboxes
   - Stratification checkboxes filter visible frame cards
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

    // Helper: collect currently checked strata filters
    function getSelectedStrata() {
        const checked = Array.from(document.querySelectorAll(".strata-filter"))
            .filter(cb => cb.checked)
            .map(cb => cb.value);
        return new Set(checked);
    }

    // Update visible frame cards based on selected strata filters.
    // Show a card if any of its strata values match the selected set.
    // If no filters are selected, show NO cards.
    function updateFrameVisibility() {
        const selected = getSelectedStrata();
        frameCards.forEach(card => {
            const strataRaw = card.getAttribute("data-strata") || "";
            const strataList = strataRaw.split(",").map(s => s.trim()).filter(Boolean);
            let visible = false;
            if (selected.size > 0) {
                visible = strataList.some(s => selected.has(s));
            }
            card.style.display = visible ? "" : "none";
        });
    }

    // Wire change handlers for each strata checkbox to update visibility
    const strataCheckboxes = document.querySelectorAll(".strata-filter");
    strataCheckboxes.forEach(cb => {
        cb.addEventListener("change", () => {
            updateFrameVisibility();
            // update select-all checkbox state if present
            const selectAll = document.getElementById("select-all-strata");
            if (selectAll) {
                const all = Array.from(strataCheckboxes);
                selectAll.checked = all.every(x => x.checked);
                selectAll.indeterminate = !selectAll.checked && all.some(x => x.checked);
            }
        });
    });

    // Select All for stratification filters
    const selectAllStrata = document.getElementById("select-all-strata");
    if (selectAllStrata) {
        selectAllStrata.addEventListener("change", (e) => {
            const checked = e.target.checked;
            strataCheckboxes.forEach((cb) => {
                cb.checked = checked;
            });
            updateFrameVisibility();
            selectAllStrata.indeterminate = false;
        });
        // initialize select-all state
        const all = Array.from(strataCheckboxes);
        if (all.length > 0) {
            selectAllStrata.checked = all.every(x => x.checked);
            selectAllStrata.indeterminate = !selectAllStrata.checked && all.some(x => x.checked);
        }
    }

    // Initial visibility update (respect initial checkbox states)
    updateFrameVisibility();
});
