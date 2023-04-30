# Documentation

これは、ドキュメントを編集する方法と、適切に貢献する方法を示し、私たちにいくつかのルールを概説します。

## How to edit

すべてのドキュメントはMarkdownで書かれており、`docs`フォルダにあります。GitHubで直接編集することもできますし、リポジトリをクローンしてローカルで編集することもできます。

GitHubでの編集は、変更を伴うプルリクエストを作成し、レビュー待ちになります。

一度変更がレビューされ、承認されると、ブランチにマージされ、CI/CDパイプラインによってデプロイされます。


## Running documentation locally

::: info
`pnpm` can be installed using `npm install -g pnpm`
:::

リポジトリをクローン

```bash
git clone https://github.com/ddPn08/Radiata.git
```

依存関係のインストール

```bash
pnpm i
```

開発用サーバーを実行

```bash
pnpm docs:dev
```

`http://localhost:5173/Radiata/`でドキュメントにアクセスできます。
