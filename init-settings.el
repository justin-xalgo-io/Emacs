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

;; Private methods.
(defun insert-char-4 ()
  "Read 4 keyboard inputs, interpret it as a hexadecimal number, and insert it as a character."
  (interactive)
  (let* ((k1 (read-key-sequence "____"))
         (k2 (read-key-sequence (concat k1 "___")))
         (k3 (read-key-sequence (concat k1 k2 "__")))
         (k4 (read-key-sequence (concat k1 k2 k3 "_")))
         (charcode (cl-parse-integer (concat k1 k2 k3 k4) :radix 16)))
    (insert-char charcode)
    (message (concat k1 k2 k3 k4 " => " (char-to-string charcode)))))

(defun insert-v-bar ()
  (interactive)
  (insert-char (cl-parse-integer "007C" :radix 16)))

