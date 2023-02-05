/**  @type {import('eslint').Linter.Config} */
module.exports = {
  env: {
    browser: true,
    node: true,
  },
  extends: [
    'plugin:solid/recommended',
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:import/recommended',
    'plugin:import/typescript',
    'prettier',
  ],
  plugins: ['unused-imports', 'solid'],
  rules: {
    'no-empty': 'off',
    'solid/reactivity': 'off',
    '@typescript-eslint/ban-types': 'off',
    '@typescript-eslint/no-var-requires': 'off',
    '@typescript-eslint/no-empty-interface': 'off',
    '@typescript-eslint/no-unused-vars': 'off',
    '@typescript-eslint/no-explicit-any': 'off',
    '@typescript-eslint/no-non-null-assertion': 'off',
    '@typescript-eslint/no-empty-function': 'off',
    '@typescript-eslint/no-namespace': 'off',
    '@typescript-eslint/no-non-null-asserted-optional-chain': 'off',
    'import/no-unresolved': 'off',
    'unused-imports/no-unused-imports': 'error',
    'unused-imports/no-unused-vars': [
      'warn',
      {
        vars: 'all',
        varsIgnorePattern: '^_',
        args: 'all',
        argsIgnorePattern: '^_',
      },
    ],
    'import/order': [
      'error',
      {
        alphabetize: {
          order: 'asc',
        },
        groups: [['builtin', 'external', 'internal'], ['parent', 'sibling', 'index'], ['object']],
        'newlines-between': 'always',
      },
    ],
  },
}
