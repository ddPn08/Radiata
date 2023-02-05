import autosize from 'autosize'
import { css, styled } from 'decorock'
import type { GenerateImageRequest } from 'internal:api'
import { createSignal, onMount, Show } from 'solid-js'
import { createStore } from 'solid-js/store'

import { api } from '~/api'
import { Gallery } from '~/components/gallery'
import { Button } from '~/components/ui/button'
import { Grid } from '~/components/ui/grid'
import { Input } from '~/components/ui/input'
import { Select } from '~/components/ui/select'
import { WithSlider } from '~/components/ui/slider'
import { VStack } from '~/components/ui/stack'
import { Textarea } from '~/components/ui/textarea'

const Container = styled.div`
  textarea {
    resize: none;
  }
`

const Item = styled.div`
  width: 100%;
`

export const Txt2Img = () => {
  const [promptRef, setPromptRef] = createSignal<HTMLTextAreaElement>()
  const [npromptRef, setNPromptRef] = createSignal<HTMLTextAreaElement>()
  const [req, setReq] = createStore({
    prompt: 'masterpiece, best quality, 1girl',
    negative_prompt:
      'worst quality, low quality, normal quality, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry',
    image_height: 512,
    image_width: 512,
    scheduler_id: 'euler_a',
    scale: 7.5,
    batch_count: 1,
    steps: 20,
    seed: -1,
  } as Required<GenerateImageRequest>)
  const [images, setImages] = createSignal<[string, Record<string, string>][]>([])
  const [time, setTime] = createSignal<number | null>(null)

  onMount(() => {
    autosize(promptRef()!)
    autosize(npromptRef()!)
  })

  return (
    <Container>
      <Grid
        class={css`
          grid-template-columns: 4fr 1fr;
          grid-template-rows: 100%;
        `}
      >
        <VStack>
          <Textarea
            ref={setPromptRef}
            placeholder="Prompt"
            value={req.prompt}
            onInput={(e) => setReq('prompt', e.currentTarget.value)}
          />
          <Textarea
            ref={setNPromptRef}
            placeholder="Negative Prompt"
            value={req.negative_prompt}
            onInput={(e) => setReq('negative_prompt', e.currentTarget.value)}
          />
        </VStack>
        <Button
          // eslint-disable-next-line solid/reactivity
          task={async () => {
            const res = await api.generateImage({ generateImageRequest: { ...req } })
            setImages(Object.entries(res.data.images) as any)
            setTime(res.data.performance)
          }}
        >
          Generate
        </Button>
      </Grid>
      <br />
      <Grid
        class={css`
          justify-content: space-between;
          gap: 1rem;
          grid-template-columns: 3fr 2.25fr;
          grid-template-rows: 100%;
        `}
      >
        <VStack
          class={css`
            gap: 1rem;
          `}
          inline
        >
          <div
            class={css`
              display: flex;
              width: 100%;
              align-items: center;
              gap: 1rem;
            `}
          >
            <div>Sampler</div>
            <Select
              options={['ddim', 'euler', 'euler_a', 'pndm'].map((v) => ({ label: v, value: v }))}
              value={req.scheduler_id}
              onChange={(opt) => setReq('scheduler_id', opt.value)}
            />
          </div>
          <WithSlider
            label="steps"
            max={1000}
            min={1}
            step={1}
            value={req.steps}
            onChange={(e) => setReq('steps', parseInt(e))}
          />
          <Grid
            class={css`
              gap: 0.5rem;
              grid-template-columns: 3fr 2fr;
              grid-template-rows: 100%;
            `}
          >
            <VStack inline>
              <WithSlider
                label="height"
                max={1024}
                min={256}
                step={8}
                value={req.image_height}
                onChange={(e) => setReq('image_height', parseInt(e))}
              />
              <WithSlider
                label="width"
                max={1024}
                min={256}
                step={8}
                value={req.image_width}
                onChange={(e) => setReq('image_width', parseInt(e))}
              />
            </VStack>
            <VStack inline>
              <WithSlider
                label="Batch count"
                max={100}
                min={0}
                step={1}
                value={req.batch_count}
                onChange={(v) => setReq('batch_count', parseFloat(v))}
              />
              <WithSlider
                label="CFG Scale"
                max={20}
                min={0}
                step={0.5}
                value={req.scale}
                onChange={(v) => setReq('scale', parseFloat(v))}
              />
            </VStack>
          </Grid>

          <Item>
            <div>Seed</div>
            <Input
              type="number"
              value={req.seed}
              onInput={(e) => setReq('seed', parseInt(e.currentTarget.value))}
            />
          </Item>
        </VStack>
        <div>
          <Gallery images={images()} />
          <Show when={time()} keyed>
            {(time) => <div>{Math.round(time) / 1000}s</div>}
          </Show>
        </div>
      </Grid>
    </Container>
  )
}
