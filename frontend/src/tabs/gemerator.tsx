import { Box, Button, Divider, Flex, Image, MediaQuery, Stack, Textarea } from '@mantine/core'
import { useAtom } from 'jotai'
import { GenerationParamertersForm, generationParametersAtom } from '~/atoms/generationParameters'
import Parameters from '~/components/parameters'

const Generator = () => {
  const [parameters, setParameters] = useAtom(generationParametersAtom)

  const onModelChange = (value: string) => {}

  const onSubmit = (values: GenerationParamertersForm) => {
    console.log(values)
  }

  return (
    <Box h={'100%'}>
      <form
        style={{
          height: '100%',
        }}
        onSubmit={(e) => {
          e.preventDefault()
          onSubmit(parameters)
        }}
      >
        <Flex h={'100%'}>
          <Stack w={'100%'} m={'md'}>
            <Stack w={'100%'}>
              <Textarea
                label={'Positive'}
                defaultValue={parameters.prompt}
                onChange={(e) => {
                  setParameters((p) => ({
                    ...p,
                    prompt: e.target.value,
                  }))
                }}
                autosize
              />
              <Textarea
                label={'Negative'}
                defaultValue={parameters.negative_prompt}
                onChange={(e) => setParameters((p) => ({ ...p, negative_prompt: e.target.value }))}
                autosize
              />
            </Stack>

            <Button type={'submit'}>Generate</Button>

            <Box>
              <Image />
            </Box>
          </Stack>

          <Divider orientation="vertical" />

          <MediaQuery
            smallerThan={'sm'}
            styles={{
              display: 'none',
            }}
          >
            <Parameters />
          </MediaQuery>
        </Flex>
      </form>
    </Box>
  )
}

export default Generator
