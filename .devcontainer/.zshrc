# .zshrc

# Enable colors
autoload -U colors && colors

# Set prompt
PROMPT="%F{green}%n@%m%f:%F{blue}%~%f$ "

# Enable syntax highlighting
# source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh

# Add aliases
alias ll="ls -la"
alias vim="vim -O"

# Powerlevel10k configuration (optional)
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh