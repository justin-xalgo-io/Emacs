(require 'package)

(add-to-list 'package-archives
             '("elpy" . "http://jorgenschaefer.github.io/packages/"))

(add-to-list 'package-archives
             '("marmalade" . "http://marmalade-repo.org/packages/"))

(add-to-list 'package-archives
             '("melpa" . "http://melpa.org/packages/") t)

(add-to-list 'load-path "~/.emacs.d/site-lisp/")

; List required packages.
(setq package-list
    '(python-environment deferred epc 
        flycheck ctable jedi concurrent company cyberpunk-theme elpy 
        yasnippet pyvenv highlight-indentation find-file-in-project 
        sql-indent sql exec-path-from-shell iedit projectile auctex
        auto-complete popup let-alist magit minimap popup rust-mode
		python-mode doom-themes))

; Activate all the packages.
(package-initialize)

; Fetch the list of available packages. 
(unless package-archive-contents
  (package-refresh-contents))

; Install any missing packages.
(dolist (package package-list)
  (unless (package-installed-p package)
    (package-install package)))
