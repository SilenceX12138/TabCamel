# Remove all local branches except master
git branch | grep -v "master" | xargs git branch -D
# Remove all remote branches except master
git fetch --prune
git branch -r | grep -v " -> " | grep "origin/" | grep -v "origin/master$" | sed 's/^\s*origin\///' | xargs -I {} git push origin --delete {}
git branch -r | grep -v " -> " | grep "public/" | grep -v "public/master$" | sed 's/^\s*public\///' | xargs -I {} git push public --delete {}