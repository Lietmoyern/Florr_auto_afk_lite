// ==UserScript==
// @name Opaque Everything Toggle with Local Control
// @namespace http://tampermonkey.net/
// @version 3.1
// @description Controls transparency via local Python script. Default state is OFF.
// @author m0m0m0746
// @include https://zorr.pro/*
// @include https://florr.io/*
// @grant GM_xmlhttpRequest
// @run-at document-start
// @connect localhost
// ==/UserScript==

(function() {
    'use strict';

    let blockAlpha = false;
    const originalStyles = new WeakMap();
    const processedContexts = new WeakSet();
    const processedElements = new WeakSet();
    let ws = null;
    let isConnected = false;
    let statusDisplay = null;
    let connectButton = null;
    let container = null;
    let autoConnectAttempted = false;

    // --- 1. Canvas Handling Functions --- 
    function makeSemiTransparentOpaque(color) {
        if (!blockAlpha || typeof color !== 'string') return color;

        let modifiedColor = color.replace(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)/gi, (match, r, g, b, a) => {
            const alpha = parseFloat(a);
            return alpha > 0 && alpha < 1 ? `rgba(${r},${g},${b},1)` : match;
        });

        modifiedColor = modifiedColor.replace(/hsla\((\d+),\s*([\d.]+)%?,\s*([\d.]+)%?,\s*([\d.]+)\)/gi, (match, h, s, l, a) => {
            const alpha = parseFloat(a);
            return alpha > 0 && alpha < 1 ? `hsla(${h},${s}%,${l}%,1)` : match;
        });

        return modifiedColor;
    }

    function enforceSmartOpaqueDrawing(ctx) {
        if (processedContexts.has(ctx)) return;
        processedContexts.add(ctx);

        const originalFillStyleDescriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(ctx), 'fillStyle');
        if (originalFillStyleDescriptor?.set) {
            const setter = originalFillStyleDescriptor.set;
            Object.defineProperty(ctx, 'fillStyle', {
                set: function(value) {
                    setter.call(this, blockAlpha ? makeSemiTransparentOpaque(value) : value);
                },
                get: originalFillStyleDescriptor.get,
                configurable: true
            });
        }

        const originalStrokeStyleDescriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(ctx), 'strokeStyle');
        if (originalStrokeStyleDescriptor?.set) {
            const setter = originalStrokeStyleDescriptor.set;
            Object.defineProperty(ctx, 'strokeStyle', {
                set: function(value) {
                    setter.call(this, blockAlpha ? makeSemiTransparentOpaque(value) : value);
                },
                get: originalStrokeStyleDescriptor.get,
                configurable: true
            });
        }

        const originalGlobalAlphaDescriptor = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(ctx), 'globalAlpha');
        if (originalGlobalAlphaDescriptor?.set) {
            const setter = originalGlobalAlphaDescriptor.set;
            Object.defineProperty(ctx, 'globalAlpha', {
                set: function(value) {
                    setter.call(this, blockAlpha && value > 0 && value < 1 ? 1 : value);
                },
                get: originalGlobalAlphaDescriptor.get,
                configurable: true
            });
            if (blockAlpha && ctx.globalAlpha > 0 && ctx.globalAlpha < 1) ctx.globalAlpha = 1;
        }
    }

    // --- 2. DOM Element Handling Functions ---
    const colorProperties = [
        'backgroundColor', 'borderColor', 'borderTopColor',
        'borderRightColor', 'borderBottomColor', 'borderLeftColor',
        'color', 'outlineColor', 'textDecorationColor', 'backgroundImage'
    ];

    function applyOpaqueStyles(element) {
        if (processedElements.has(element)) return;
        processedElements.add(element);

        const computedStyle = getComputedStyle(element);
        const currentOriginalStyles = {};

        if (!currentOriginalStyles.hasOwnProperty('opacity')) {
            currentOriginalStyles.opacity = computedStyle.opacity;
        }

        const opacity = parseFloat(computedStyle.opacity);
        if (opacity > 0 && opacity < 1) {
            element.style.opacity = '1';
        } else if (opacity === 0) {
            currentOriginalStyles.wasTransparent = true;
        }

        for (const prop of colorProperties) {
            const value = computedStyle[prop];
            if (value === 'rgba(0, 0, 0, 0)' || value === 'transparent') {
                currentOriginalStyles[prop] = value;
                continue;
            }

            const newValue = makeSemiTransparentOpaque(value);
            if (newValue !== value) {
                currentOriginalStyles[prop] = element.style[prop] || '';
                element.style[prop] = newValue;
            }
        }

        const borderProps = ['borderTopColor', 'borderRightColor',
                            'borderBottomColor', 'borderLeftColor'];
        borderProps.forEach(prop => {
            const value = computedStyle[prop];
            if (value === 'rgba(0, 0, 0, 0)' || value === 'transparent') {
                currentOriginalStyles[prop] = value;
                element.style[prop] = 'transparent';
            }
        });

        originalStyles.set(element, currentOriginalStyles);
    }

    function revertOriginalStyles(element) {
        if (!originalStyles.has(element)) return;

        const storedStyles = originalStyles.get(element);
        if (storedStyles.wasTransparent) {
            element.style.opacity = '0';
        } else {
            element.style.opacity = storedStyles.opacity || '';
        }

        for (const prop of colorProperties) {
            if (prop in storedStyles) {
                if (storedStyles[prop] === 'transparent' ||
                    storedStyles[prop] === 'rgba(0, 0, 0, 0)') {
                    element.style[prop] = 'transparent';
                } else {
                    element.style[prop] = storedStyles[prop];
                }
            }
        }

        processedElements.delete(element);
    }

    function processNodeAndChildren(node) {
        if (node.nodeType === Node.ELEMENT_NODE) {
            if (blockAlpha) {
                applyOpaqueStyles(node);
            } else {
                revertOriginalStyles(node);
            }
            for (const child of node.children) {
                processNodeAndChildren(child);
            }
        }
    }

    // --- 3. DOM Monitor - Listen for new elements ---
    const observer = new MutationObserver(mutations => {
        if (!blockAlpha) return;
        mutations.forEach(mutation => {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType === 1) {
                    processNodeAndChildren(node);
                }
            });
        });
    });

    // --- 4. Connection Management Functions --- 
    function createConnectionUI() {
        container = document.createElement('div');
        container.id = 'opaque-control-container';
        container.style.position = 'fixed';
        container.style.top = '10px';
        container.style.right = '10px';
        container.style.zIndex = '99999';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.gap = '5px';
        container.style.alignItems = 'flex-end';
        container.style.opacity = '0'; 
        container.style.transition = 'opacity 0.3s ease';

        
        const hoverArea = document.createElement('div');
        hoverArea.id = 'opaque-hover-area';
        hoverArea.style.position = 'fixed';
        hoverArea.style.top = '0';
        hoverArea.style.right = '0';
        hoverArea.style.width = '100px';
        hoverArea.style.height = '50px';
        hoverArea.style.zIndex = '99998';
        hoverArea.style.cursor = 'default';

        hoverArea.addEventListener('mouseenter', () => {
            if (container) {
                container.style.opacity = '1';
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (container && e.clientX > window.innerWidth - 150 && e.clientY < 100) {
                container.style.opacity = '1';
            } else if (container && isConnected) {
                container.style.opacity = '0';
            }
        });

        statusDisplay = document.createElement('div');
        statusDisplay.id = 'opaque-status';
        statusDisplay.textContent = 'Disconnected';
        statusDisplay.style.padding = '5px 10px';
        statusDisplay.style.backgroundColor = '#f44336';
        statusDisplay.style.color = 'white';
        statusDisplay.style.borderRadius = '5px';
        statusDisplay.style.fontSize = '12px';
        statusDisplay.style.fontWeight = 'bold';

        connectButton = document.createElement('button');
        connectButton.id = 'opaque-connect-btn';
        connectButton.textContent = 'Connect to Local Script';
        connectButton.style.padding = '8px 12px';
        connectButton.style.backgroundColor = '#2196F3';
        connectButton.style.color = 'white';
        connectButton.style.border = 'none';
        connectButton.style.borderRadius = '5px';
        connectButton.style.cursor = 'pointer';
        connectButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
        connectButton.style.fontSize = '14px';

        connectButton.addEventListener('click', () => {
            connectToLocal();
        });

        container.appendChild(statusDisplay);
        container.appendChild(connectButton);
        document.body.appendChild(container);
        document.body.appendChild(hoverArea);
    }

    function updateStatus(message, color) {
        if (statusDisplay) {
            statusDisplay.textContent = message;
            statusDisplay.style.backgroundColor = color;
        }
    }

    function connectToLocal() {
        if (isConnected) return;

        updateStatus('Connecting...', '#FF9800');

        try {
            ws = new WebSocket('ws://localhost:8765');

            ws.onopen = function() {
                isConnected = true;
                updateStatus('Connected', '#4CAF50');
                
                if (connectButton) {
                    connectButton.style.display = 'none';
                }
                
                if (autoConnectAttempted) {
                    container.style.opacity = '0';
                }

                ws.send(JSON.stringify({ action: 'getState' }));
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.action === 'setState') {
                    blockAlpha = data.state;
                    applyCurrentAlphaStateToAllElements();
                }
            };

            ws.onclose = function() {
                isConnected = false;
                updateStatus('Disconnected', '#f44336');
                if (connectButton) {
                    connectButton.style.display = 'block';
                }
                if (container) {
                    container.style.opacity = '1';
                }
            };

            ws.onerror = function() {
                isConnected = false;
                updateStatus('Connection Error', '#f44336');
                if (connectButton) {
                    connectButton.style.display = 'block';
                }
            };
        } catch (error) {
            console.error('WebSocket connection error:', error);
            updateStatus('Connection Failed', '#f44336');
        }
    }

    function applyCurrentAlphaStateToAllElements() {
        const allElements = document.querySelectorAll('*');
        allElements.forEach(element => {
            if (blockAlpha) {
                applyOpaqueStyles(element);
            } else {
                revertOriginalStyles(element);
            }
        });

        const allCanvases = document.querySelectorAll('canvas');
        allCanvases.forEach(canvas => {
            const ctx = canvas.getContext('2d');
            if (ctx) {
                // Reset context properties
                ctx.globalAlpha = blockAlpha ? 1 : 1; // Actual value handled by setter hooks
            }
        });
    }

    // --- 5. Initialization and Integration --- 
    document.addEventListener('DOMContentLoaded', () => {
        const originalGetContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(contextType, ...args) {
            const ctx = originalGetContext.apply(this, [contextType, ...args]);
            if (contextType === '2d' && ctx) enforceSmartOpaqueDrawing(ctx);
            return ctx;
        };

        createConnectionUI();
        

        autoConnectAttempted = true;
        setTimeout(() => {
            connectToLocal();
        }, 1000); 

        observer.observe(document, { childList: true, subtree: true });
    });
})();