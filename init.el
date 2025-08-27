;; Load packages.
(load "~/.emacs.d/init-packages")

;; Load personal settings.
(load "~/.emacs.d/init-settings")

(use-package doom-themes
  :ensure t
  :custom

  ;; Global settings (defaults).
  (doom-themes-enable-bold t)   ; if nil, bold is universally disabled
  (doom-themes-enable-italic t) ; if nil, italics is universally disabled

  ;; For treemacs users.
  (doom-themes-treemacs-theme "doom-atom") ; use "doom-colors" for less minimal icon theme

  :config
  (load-theme 'doom-bluloco-dark t)

  ;; Enable flashing mode-line on errors.
  (doom-themes-visual-bell-config)

  ;; Enable custom neotree theme (nerd-icons must be installed!)
  (doom-themes-neotree-config)

  ;; or for treemacs users.
  (doom-themes-treemacs-config)

  ;; Corrects (and improves) org-mode's native fontification.
  (doom-themes-org-config))

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(custom-safe-themes
	 '("22a0d47fe2e6159e2f15449fcb90bbf2fe1940b185ff143995cc604ead1ea171"
		 "014cb63097fc7dbda3edf53eb09802237961cbb4c9e9abd705f23b86511b0a69"
		 "f053f92735d6d238461da8512b9c071a5ce3b9d972501f7a5e6682a90bf29725"
		 "02d422e5b99f54bd4516d4157060b874d14552fe613ea7047c4a5cfa1288cf4f"
		 "a6920ee8b55c441ada9a19a44e9048be3bfb1338d06fc41bce3819ac22e4b5a1"
		 "e14289199861a5db890065fdc5f3d3c22c5bac607e0dbce7f35ce60e6b55fc52"
		 "e8bd9bbf6506afca133125b0be48b1f033b1c8647c628652ab7a2fe065c10ef0"
		 "b754d3a03c34cfba9ad7991380d26984ebd0761925773530e24d8dd8b6894738"
		 "3061706fa92759264751c64950df09b285e3a2d3a9db771e99bcbb2f9b470037"
		 "f64189544da6f16bab285747d04a92bd57c7e7813d8c24c30f382f087d460a33"
		 "0c83e0b50946e39e237769ad368a08f2cd1c854ccbcd1a01d39fdce4d6f86478"
		 "56044c5a9cc45b6ec45c0eb28df100d3f0a576f18eef33ff8ff5d32bac2d9700"
		 "0f1341c0096825b1e5d8f2ed90996025a0d013a0978677956a9e61408fcd2c77"
		 "0d2c5679b6d087686dcfd4d7e57ed8e8aedcccc7f1a478cd69704c02e4ee36fe"
		 "77fff78cc13a2ff41ad0a8ba2f09e8efd3c7e16be20725606c095f9a19c24d3d"
		 "4594d6b9753691142f02e67b8eb0fda7d12f6cc9f1299a49b819312d6addad1d"
		 "1f292969fc19ba45fbc6542ed54e58ab5ad3dbe41b70d8cb2d1f85c22d07e518"
		 "f1e8339b04aef8f145dd4782d03499d9d716fdc0361319411ac2efc603249326"
		 "e1df746a4fa8ab920aafb96c39cd0ab0f1bac558eff34532f453bd32c687b9d6"
		 "4990532659bb6a285fee01ede3dfa1b1bdf302c5c3c8de9fad9b6bc63a9252f7"
		 "ff24d14f5f7d355f47d53fd016565ed128bf3af30eb7ce8cae307ee4fe7f3fd0"
		 "df6dfd55673f40364b1970440f0b0cb8ba7149282cf415b81aaad2d98b0f0290"
		 default))
 '(package-selected-packages
	 '(## auctex company cyberpunk-theme doom-themes elpy epc
				exec-path-from-shell find-file-in-project flycheck
				git-rebase-mode iedit jedi magit minimap projectile
				python-environment python-mode rust-mode sql-indent)))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 )
