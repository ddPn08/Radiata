import { ActionIcon, Flex, Input } from '@mantine/core'
import { IconDice5 } from '@tabler/icons-react'
import { useAtom } from 'jotai'
import { generationParametersAtom } from '~/atoms/generationParameters'
import BetterNumInput from '~/components/betterNumInput'
import { generateRandomNumber } from '~/utils/rand'

const SeedParameter = () => {
  const [parameters, setParameters] = useAtom(generationParametersAtom)

  return (
    <Input.Wrapper label={'Seed'}>
      <Flex w={'100%'} align={'center'} gap={'sm'}>
        <BetterNumInput
          defaultValue={parameters.seed}
          value={parameters.seed}
          min={-1}
          max={4294967295}
          step={1}
          onChange={(e) => {
            if (e) {
              setParameters((p) => ({ ...p, seed: e }))
            }
          }}
          w={'100%'}
          allowWheel
        />
        <ActionIcon
          variant={'outline'}
          color={'blue'}
          onClick={() => {
            setParameters((p) => ({
              ...p,
              seed: generateRandomNumber(0, 4294967295),
            }))
          }}
        >
          <IconDice5 size={16} />
        </ActionIcon>
      </Flex>
    </Input.Wrapper>
  )
}

export default SeedParameter
