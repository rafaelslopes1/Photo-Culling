/**
 * 🎯 Photo Culling Web App v2.0 - Expert Evaluation Interface
 * JavaScript for interactive expert evaluation system
 */

class ExpertEvaluationApp {
    constructor() {
        this.currentImageIndex = 0;
        this.images = [];
        this.evaluationData = {
            ratings: {},
            categorical_assessments: {},
            decisions: {},
            issues: {},
            confidence_level: 50,
            evaluation_time_seconds: 0,
            comments: ''
        };
        this.startTime = null;
        this.totalImages = 0;
        this.evaluatedImages = 0;
        
        // Initialize pan state for image zoom and pan
        this.panState = {
            isZoomed: false,
            isPanning: false,
            startX: 0,
            startY: 0,
            currentX: 0,
            currentY: 0,
            scale: 1
        };
        
        this.init();
    }

    async init() {
        // Load available images
        await this.loadImages();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Setup image pan functionality
        this.setupImagePan();
        
        // Load first image
        if (this.images.length > 0) {
            this.loadImage(0);
        }
        
        // Update progress
        this.updateProgress();
        
        console.log('🎯 Expert Evaluation App initialized');
    }

    async loadImages() {
        try {
            const response = await fetch('/api/images');
            const data = await response.json();
            
            if (data.success) {
                this.images = data.images;
                this.totalImages = this.images.length;
                console.log(`📸 Loaded ${this.totalImages} images for evaluation`);
            } else {
                this.showMessage('Erro ao carregar imagens', 'error');
            }
        } catch (error) {
            console.error('Error loading images:', error);
            this.showMessage('Erro de conexão ao carregar imagens', 'error');
        }
    }

    loadImage(index) {
        if (index < 0 || index >= this.images.length) return;
        
        this.currentImageIndex = index;
        const imageData = this.images[index];
        
        // Reset evaluation data
        this.resetEvaluationData();
        
        // Update UI
        document.getElementById('current-image').src = `/api/image/${imageData.filename}`;
        document.getElementById('image-filename').textContent = imageData.filename;
        document.getElementById('image-counter').textContent = 
            `${index + 1} de ${this.images.length}`;
        
        // Reset zoom when loading new image
        this.resetImageZoom();
        
        // Start timer
        this.startTime = Date.now();
        
        // Update progress
        this.updateProgress();
        
        console.log(`🖼️ Loaded image: ${imageData.filename}`);
    }

    resetEvaluationData() {
        this.evaluationData = {
            ratings: {},
            categorical_assessments: {},
            decisions: {},
            issues: {},
            confidence_level: 50,
            evaluation_time_seconds: 0,
            comments: ''
        };
        
        // Reset UI elements
        document.querySelectorAll('.star').forEach(star => {
            star.classList.remove('filled');
        });
        
        document.querySelectorAll('.cat-btn').forEach(button => {
            button.classList.remove('active');
        });
        
        document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = false;
        });
        
        document.getElementById('confidence-slider').value = 50;
        document.getElementById('confidence-value').textContent = '50%';
        document.getElementById('comments').value = '';
    }

    setupEventListeners() {
        // Star ratings
        document.querySelectorAll('.star-rating').forEach(ratingGroup => {
            const ratingName = ratingGroup.dataset.rating;
            const stars = ratingGroup.querySelectorAll('.star');
            
            stars.forEach((star, index) => {
                star.addEventListener('click', () => {
                    this.setRating(ratingName, index + 1);
                });
                
                star.addEventListener('mouseenter', () => {
                    this.highlightStars(stars, index + 1);
                });
            });
            
            ratingGroup.addEventListener('mouseleave', () => {
                this.updateStarDisplay(stars, this.evaluationData.ratings[ratingName] || 0);
            });
        });

        // Decision toggles
        document.querySelectorAll('.decision-toggle').forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const decisionName = e.target.dataset.decision;
                this.evaluationData.decisions[decisionName] = e.target.checked;
            });
        });

        // Confidence slider
        const confidenceSlider = document.getElementById('confidence-slider');
        if (confidenceSlider) {
            confidenceSlider.addEventListener('input', (e) => {
                const value = e.target.value;
                this.evaluationData.confidence_level = parseInt(value);
                document.getElementById('confidence-value').textContent = `${value}%`;
            });
        }

        // Comments
        const commentsField = document.getElementById('comments');
        if (commentsField) {
            commentsField.addEventListener('input', (e) => {
                this.evaluationData.comments = e.target.value;
            });
        }

        // Categorical buttons
        document.querySelectorAll('.categorical-buttons').forEach(buttonGroup => {
            const category = buttonGroup.dataset.category;
            const buttons = buttonGroup.querySelectorAll('.cat-btn');
            
            buttons.forEach(button => {
                button.addEventListener('click', () => {
                    this.setCategoricalValue(category, button.dataset.value, buttonGroup);
                });
            });
        });

        // Technical issues checkboxes
        document.querySelectorAll('.issue-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                this.updateTechnicalIssues();
            });
        });

        // Navigation buttons
        document.getElementById('prev-btn')?.addEventListener('click', () => {
            this.previousImage();
        });
        
        document.getElementById('next-btn')?.addEventListener('click', () => {
            this.nextImage();
        });

        // Action buttons
        document.getElementById('submit-btn')?.addEventListener('click', () => {
            this.submitEvaluation();
        });
        
        document.getElementById('quick-reject-btn')?.addEventListener('click', () => {
            this.quickReject();
        });
        
        document.getElementById('skip-btn')?.addEventListener('click', () => {
            this.skipImage();
        });

        // Image controls - NO CLICK LISTENER HERE (handled in setupImagePan)
        document.getElementById('zoom-fit-btn')?.addEventListener('click', () => {
            this.resetImageZoom();
        });
        
        document.getElementById('zoom-100-btn')?.addEventListener('click', () => {
            this.setImageZoom100();
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ignore if user is typing in text field
            if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {
                return;
            }

            switch (e.key) {
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                    this.setRating('overall_quality', parseInt(e.key));
                    break;
                case 'q':
                case 'Q':
                    this.quickReject();
                    break;
                case ' ':  // Spacebar
                    e.preventDefault();
                    this.nextImage();
                    break;
                case 'ArrowLeft':
                    this.previousImage();
                    break;
                case 'ArrowRight':
                    this.nextImage();
                    break;
                case 'z':
                case 'Z':
                    this.resetImageZoom();
                    break;
                case 'x':
                case 'X':
                    this.setImageZoom100();
                    break;
                case 'Enter':
                    if (e.ctrlKey || e.metaKey) {
                        this.submitEvaluation();
                    }
                    break;
                case '?':
                    this.toggleShortcutsHelp();
                    break;
            }
        });
    }

    setRating(ratingName, value) {
        this.evaluationData.ratings[ratingName] = value;
        
        // Update star display
        const ratingGroup = document.querySelector(`[data-rating="${ratingName}"]`);
        if (ratingGroup) {
            const stars = ratingGroup.querySelectorAll('.star');
            this.updateStarDisplay(stars, value);
        }
        
        console.log(`⭐ Rating ${ratingName}: ${value}`);
    }

    highlightStars(stars, count) {
        stars.forEach((star, index) => {
            if (index < count) {
                star.classList.add('filled');
            } else {
                star.classList.remove('filled');
            }
        });
    }

    updateStarDisplay(stars, rating) {
        stars.forEach((star, index) => {
            if (index < rating) {
                star.classList.add('filled');
            } else {
                star.classList.remove('filled');
            }
        });
    }

    setCategoricalValue(category, value, buttonGroup) {
        // Initialize categorical assessments if not exists
        if (!this.evaluationData.categorical_assessments) {
            this.evaluationData.categorical_assessments = {};
        }
        
        // Set the value
        this.evaluationData.categorical_assessments[category] = value;
        
        // Update button states
        const buttons = buttonGroup.querySelectorAll('.cat-btn');
        buttons.forEach(btn => {
            btn.classList.remove('active');
        });
        
        const selectedButton = buttonGroup.querySelector(`[data-value="${value}"]`);
        if (selectedButton) {
            selectedButton.classList.add('active');
        }
        
        console.log(`📋 Categorical ${category}: ${value}`);
    }

    updateTechnicalIssues() {
        const selectedIssues = [];
        document.querySelectorAll('.issue-checkbox:checked').forEach(checkbox => {
            selectedIssues.push(checkbox.dataset.issue);
        });
        
        // Initialize categorical assessments if not exists
        if (!this.evaluationData.categorical_assessments) {
            this.evaluationData.categorical_assessments = {};
        }
        
        this.evaluationData.categorical_assessments.technical_issues = selectedIssues;
        console.log(`🔧 Technical issues: ${selectedIssues.join(', ')}`);
    }

    async submitEvaluation() {
        if (!this.validateEvaluation()) {
            this.showMessage('Por favor, complete a avaliação antes de enviar', 'error');
            return;
        }

        // Calculate evaluation time
        this.evaluationData.evaluation_time_seconds = 
            Math.round((Date.now() - this.startTime) / 1000);

        // Get current image filename
        const currentImage = this.images[this.currentImageIndex];
        const submissionData = {
            image_filename: currentImage.filename,
            ...this.evaluationData
        };

        try {
            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(submissionData)
            });

            const result = await response.json();

            if (result.success) {
                this.showMessage('Avaliação salva com sucesso!', 'success');
                this.evaluatedImages++;
                this.updateProgress();
                
                // Move to next image
                setTimeout(() => {
                    this.nextImage();
                }, 1000);
            } else {
                this.showMessage('Erro ao salvar avaliação', 'error');
            }
        } catch (error) {
            console.error('Error submitting evaluation:', error);
            this.showMessage('Erro de conexão ao enviar avaliação', 'error');
        }
    }

    validateEvaluation() {
        // Check if at least overall quality is rated
        return this.evaluationData.ratings.overall_quality && 
               this.evaluationData.ratings.overall_quality > 0;
    }

    quickReject() {
        // Set quick reject values
        this.evaluationData.ratings.overall_quality = 1;
        this.evaluationData.decisions.complete_reject = true;
        this.evaluationData.confidence_level = 90;
        
        // Update UI
        this.setRating('overall_quality', 1);
        const rejectToggle = document.querySelector('[data-decision="complete_reject"]');
        if (rejectToggle) {
            rejectToggle.checked = true;
        }
        
        document.getElementById('confidence-slider').value = 90;
        document.getElementById('confidence-value').textContent = '90%';
        
        // Auto-submit
        setTimeout(() => {
            this.submitEvaluation();
        }, 500);
    }

    skipImage() {
        this.showMessage('Imagem pulada', 'warning');
        this.nextImage();
    }

    previousImage() {
        if (this.currentImageIndex > 0) {
            this.loadImage(this.currentImageIndex - 1);
        }
    }

    nextImage() {
        if (this.currentImageIndex < this.images.length - 1) {
            this.loadImage(this.currentImageIndex + 1);
        } else {
            this.showMessage('Todas as imagens avaliadas!', 'success');
            this.showCompletionStats();
        }
    }

    // Advanced image zoom and pan functionality
    setupImagePan() {
        const imageElement = document.getElementById('current-image');
        if (!imageElement) return;

        let clickTimeout = null;
        let isDragging = false;
        let dragStarted = false;

        // Click to zoom/unzoom (with delay to distinguish from drag)
        imageElement.addEventListener('click', (e) => {
            if (isDragging || dragStarted) return;
            
            clickTimeout = setTimeout(() => {
                if (!this.panState.isZoomed) {
                    // Zoom in
                    this.panState.isZoomed = true;
                    this.panState.scale = 2;
                    imageElement.classList.add('zoomed');
                    
                    // Calculate zoom position based on click coordinates
                    const rect = imageElement.getBoundingClientRect();
                    const centerX = rect.width / 2;
                    const centerY = rect.height / 2;
                    const clickX = e.clientX - rect.left;
                    const clickY = e.clientY - rect.top;
                    
                    this.panState.currentX = (centerX - clickX) * 0.5;
                    this.panState.currentY = (centerY - clickY) * 0.5;
                    
                    this.updateImageTransform();
                    imageElement.style.cursor = 'move';
                } else {
                    // Zoom out
                    this.resetImageZoom();
                }
            }, 150);
        });

        // Mouse down - start potential panning
        imageElement.addEventListener('mousedown', (e) => {
            if (!this.panState.isZoomed) return;
            
            e.preventDefault();
            clearTimeout(clickTimeout);
            
            this.panState.isPanning = true;
            dragStarted = false;
            this.panState.startX = e.clientX - this.panState.currentX;
            this.panState.startY = e.clientY - this.panState.currentY;
            
            imageElement.style.cursor = 'grabbing';
        });

        // Mouse move - panning
        document.addEventListener('mousemove', (e) => {
            if (!this.panState.isPanning) return;
            
            e.preventDefault();
            
            if (!dragStarted) {
                dragStarted = true;
                isDragging = true;
            }
            
            this.panState.currentX = e.clientX - this.panState.startX;
            this.panState.currentY = e.clientY - this.panState.startY;
            
            this.updateImageTransform();
        });

        // Mouse up - stop panning
        document.addEventListener('mouseup', (e) => {
            if (this.panState.isPanning) {
                this.panState.isPanning = false;
                imageElement.style.cursor = this.panState.isZoomed ? 'move' : 'zoom-in';
                
                // Reset drag flags after a short delay
                setTimeout(() => {
                    isDragging = false;
                    dragStarted = false;
                }, 100);
            }
        });

        // Touch support for mobile
        imageElement.addEventListener('touchstart', (e) => {
            if (e.touches.length !== 1) return;
            
            const touch = e.touches[0];
            
            if (!this.panState.isZoomed) {
                // Touch to zoom
                this.panState.isZoomed = true;
                this.panState.scale = 2;
                imageElement.classList.add('zoomed');
                
                const rect = imageElement.getBoundingClientRect();
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const touchX = touch.clientX - rect.left;
                const touchY = touch.clientY - rect.top;
                
                this.panState.currentX = (centerX - touchX) * 0.5;
                this.panState.currentY = (centerY - touchY) * 0.5;
                
                this.updateImageTransform();
            } else {
                // Start panning
                e.preventDefault();
                this.panState.isPanning = true;
                this.panState.startX = touch.clientX - this.panState.currentX;
                this.panState.startY = touch.clientY - this.panState.currentY;
            }
        });

        imageElement.addEventListener('touchmove', (e) => {
            if (!this.panState.isPanning || e.touches.length !== 1) return;
            
            e.preventDefault();
            const touch = e.touches[0];
            this.panState.currentX = touch.clientX - this.panState.startX;
            this.panState.currentY = touch.clientY - this.panState.startY;
            
            this.updateImageTransform();
        });

        imageElement.addEventListener('touchend', (e) => {
            if (this.panState.isPanning) {
                this.panState.isPanning = false;
            } else if (this.panState.isZoomed && e.touches.length === 0) {
                // Double tap to zoom out
                setTimeout(() => {
                    if (this.panState.isZoomed) {
                        this.resetImageZoom();
                    }
                }, 300);
            }
        });

        // Prevent context menu on right click
        imageElement.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });
    }

    updateImageTransform() {
        const imageElement = document.getElementById('current-image');
        if (!imageElement) return;

        const transform = `scale(${this.panState.scale}) translate(${this.panState.currentX}px, ${this.panState.currentY}px)`;
        imageElement.style.transform = transform;
    }

    toggleImageZoom() {
        const imageElement = document.getElementById('current-image');
        if (!imageElement) return;

        if (!this.panState.isZoomed) {
            // Zoom in to center
            this.panState.isZoomed = true;
            this.panState.scale = 2;
            this.panState.currentX = 0;
            this.panState.currentY = 0;
            imageElement.classList.add('zoomed');
            this.updateImageTransform();
        } else {
            this.resetImageZoom();
        }
    }

    resetImageZoom() {
        const imageElement = document.getElementById('current-image');
        if (!imageElement) return;

        this.panState.isZoomed = false;
        this.panState.scale = 1;
        this.panState.currentX = 0;
        this.panState.currentY = 0;
        this.panState.isPanning = false;

        imageElement.classList.remove('zoomed', 'panning');
        imageElement.style.transform = 'scale(1) translate(0px, 0px)';
        imageElement.style.cursor = 'zoom-in';
    }

    setImageZoom100() {
        const imageElement = document.getElementById('current-image');
        if (!imageElement) return;

        this.panState.isZoomed = true;
        this.panState.scale = 1;
        this.panState.currentX = 0;
        this.panState.currentY = 0;

        imageElement.classList.add('zoomed');
        this.updateImageTransform();
        imageElement.style.cursor = 'move';
    }

    updateProgress() {
        const progressPercent = this.totalImages > 0 ? 
            (this.evaluatedImages / this.totalImages) * 100 : 0;
        
        const progressFill = document.getElementById('progress-fill');
        if (progressFill) {
            progressFill.style.width = `${progressPercent}%`;
        }
        
        const progressText = document.getElementById('progress-text');
        if (progressText) {
            progressText.textContent = 
                `${this.evaluatedImages} de ${this.totalImages} avaliadas (${Math.round(progressPercent)}%)`;
        }
    }

    showMessage(text, type = 'info') {
        // Remove existing messages
        document.querySelectorAll('.message').forEach(msg => msg.remove());
        
        // Create new message
        const message = document.createElement('div');
        message.className = `message ${type}`;
        message.textContent = text;
        
        document.body.appendChild(message);
        
        // Show message
        setTimeout(() => {
            message.classList.add('visible');
        }, 100);
        
        // Hide message after 3 seconds
        setTimeout(() => {
            message.classList.remove('visible');
            setTimeout(() => {
                message.remove();
            }, 300);
        }, 3000);
    }

    toggleShortcutsHelp() {
        const shortcuts = document.getElementById('shortcuts-info');
        if (shortcuts) {
            shortcuts.classList.toggle('visible');
        }
    }

    showCompletionStats() {
        const avgTime = this.evaluatedImages > 0 ? 
            (Date.now() - this.sessionStartTime) / this.evaluatedImages / 1000 : 0;
        
        alert(`🎉 Sessão Concluída!\n\n` +
              `📸 Imagens avaliadas: ${this.evaluatedImages}\n` +
              `⏱️ Tempo médio por imagem: ${Math.round(avgTime)}s\n` +
              `🎯 Obrigado pela sua expertise!`);
    }

    // Session management
    startSession() {
        this.sessionStartTime = Date.now();
        this.evaluatedImages = 0;
    }

    // Export evaluation data
    async exportData() {
        try {
            const response = await fetch('/api/export');
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `expert_evaluations_${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            
            window.URL.revokeObjectURL(url);
            this.showMessage('Dados exportados com sucesso!', 'success');
        } catch (error) {
            console.error('Error exporting data:', error);
            this.showMessage('Erro ao exportar dados', 'error');
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on the evaluation page
    if (document.getElementById('current-image')) {
        window.app = new ExpertEvaluationApp();
    }
});

// Utility functions for templates
function updateConfidenceValue(value) {
    document.getElementById('confidence-value').textContent = `${value}%`;
}

// Progress tracking for multiple sessions
class SessionTracker {
    constructor() {
        this.loadSession();
    }

    loadSession() {
        const stored = localStorage.getItem('expertSession');
        if (stored) {
            this.data = JSON.parse(stored);
        } else {
            this.data = {
                totalSessions: 0,
                totalEvaluations: 0,
                averageTime: 0,
                lastSession: null
            };
        }
    }

    saveSession() {
        localStorage.setItem('expertSession', JSON.stringify(this.data));
    }

    startNewSession() {
        this.data.totalSessions++;
        this.data.lastSession = new Date().toISOString();
        this.currentSessionStart = Date.now();
        this.saveSession();
    }

    completeEvaluation(timeSeconds) {
        this.data.totalEvaluations++;
        
        // Update average time (weighted)
        if (this.data.averageTime === 0) {
            this.data.averageTime = timeSeconds;
        } else {
            this.data.averageTime = 
                (this.data.averageTime * (this.data.totalEvaluations - 1) + timeSeconds) 
                / this.data.totalEvaluations;
        }
        
        this.saveSession();
    }

    getStats() {
        return {
            ...this.data,
            sessionDuration: this.currentSessionStart ? 
                (Date.now() - this.currentSessionStart) / 1000 : 0
        };
    }
}
