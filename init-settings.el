;; Personal settings.
(setq-default indent-tabs-mode nil)
(setq-default tab-width 4)
(put 'scroll-left 'disabled nil)
(global-display-line-numbers-mode)

;; Babel settings.
(setq org-babel-python-command "python3")		; Using python3 cmd instead of python.

(org-babel-do-load-languages				 	; Load babel languages.
 'org-babel-load-languages
 '((python . t)
   (shell . t)))

;; LaTeX settings & hooks.
(add-hook 'LaTeX-mode-hook 'turn-on-reftex)  	; With AUCTeX LaTeX mode.
(add-hook 'latex-mode-hook 'turn-on-reftex)   	; With Emacs LaTeX mode.
